import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..aggregates import DecisionValidationCase
from ..entities import DecisionValidationCalibration
from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ..interfaces import (
    ConformalCalibrator,
    ConformalTransformResult,
    DecisionValidationCalibrationRepository,
    OODCalibrator,
)
from ..value_objects.gate_result import GateResult
from ..value_objects.validation_outcome import ValidationOutcome, Verdict


class Tolerance(BaseModel):
    """Tolerance thresholds used by the feasibility gate (Gate 2)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eps_l2: float | None = Field(
        default=None,
        ge=0,
        description="L2 tolerance threshold applied after adding the conformal radius.",
    )
    eps_per_obj: NDArray[np.float64] | None = Field(
        default=None,
        description="Per-object tolerance thresholds applied component-wise.",
    )

    @field_validator("eps_per_obj", mode="before")
    def _coerce_array(cls, value):  # type: ignore[override]
        if value is None:
            return value
        arr = np.asarray(value, dtype=float)
        if np.any(arr < 0):
            raise ValueError("eps_per_obj entries must be non-negative.")
        return arr

    @model_validator(mode="after")
    def _ensure_any(self):
        if self.eps_l2 is None and self.eps_per_obj is None:
            raise ValueError("Provide at least one tolerance (eps_l2 or eps_per_obj).")
        return self


class DecisionValidationService:
    """Run OOD and feasibility gates for a normalized candidate decision."""

    def __init__(
        self,
        *,
        tolerance: Tolerance,
        calibration_repository: DecisionValidationCalibrationRepository | None = None,
        calibration: DecisionValidationCalibration | None = None,
    ) -> None:
        self._tolerance = tolerance
        self._calibration_repository = calibration_repository
        self._calibration: DecisionValidationCalibration | None = None
        self._ood_calibrator: OODCalibrator | None = None
        self._conformal_calibrator: ConformalCalibrator | None = None

        if calibration is not None:
            self._apply_calibration(calibration)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_calibration(
        self,
        *,
        scope: str,
        calibration_id: str | None = None,
    ) -> DecisionValidationCalibration:
        """Load and activate a calibration bundle from the repository."""

        if self._calibration_repository is None:
            raise RuntimeError("No calibration repository configured for loading.")

        if calibration_id is not None:
            calibration = self._calibration_repository.load(
                scope=scope,
                calibration_id=calibration_id,
            )
        else:
            calibration = self._calibration_repository.load_latest(scope=scope)

        self._apply_calibration(calibration)
        return calibration

    def validate(
        self,
        *,
        candidate: NDArray[np.float64] | None = None,
        target: NDArray[np.float64] | None = None,
        **aliases: NDArray[np.float64],
    ) -> DecisionValidationCase:
        """Evaluate both assurance gates for a candidate decision."""
        if self._calibration is None:
            raise RuntimeError(
                "DecisionValidationService requires a loaded calibration before validation."
            )

        if candidate is None:
            candidate = aliases.pop("X_candidate_norm", None)
        if target is None:
            target = aliases.pop("X_target_norm", None)
        if aliases:
            unexpected = ", ".join(sorted(aliases))
            raise TypeError(f"Unexpected keyword(s) for validate(): {unexpected}")
        if candidate is None or target is None:
            raise TypeError(
                "validate() requires both 'candidate' and 'target' arrays (or their aliases)."
            )

        if self._ood_calibrator is None or self._conformal_calibrator is None:
            raise RuntimeError("Active calibration is missing required calibrators.")

        ood_metrics = self._ood_calibrator.transform(candidate)
        conformal_output: ConformalTransformResult = (
            self._conformal_calibrator.transform(candidate)
        )

        predicted = self._ensure_vector(conformal_output.prediction)
        conformal_radius = float(conformal_output.radius)

        return self._evaluate_gates(
            candidate=candidate,
            X_target=target,
            X_hat=predicted,
            ood_metrics=ood_metrics,
            conformal_radius=conformal_radius,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_gates(
        self,
        *,
        candidate: NDArray[np.float64],
        X_target: NDArray[np.float64],
        X_hat: NDArray[np.float64],
        ood_metrics: dict[str, float],
        conformal_radius: float,
    ) -> DecisionValidationCase:
        metrics: dict[str, float | bool] = {}
        explanations: dict[str, str] = {}
        gate_results: list[GateResult] = []

        gate1, gate1_metrics = self._evaluate_ood_gate(
            candidate=candidate,
            ood_metrics=ood_metrics,
        )
        gate_results.append(gate1)
        explanations[gate1.name] = gate1.explanation
        metrics.update(gate1_metrics)

        gate2, gate2_metrics = self._evaluate_tolerance_gate(
            X_target=X_target,
            X_hat=X_hat,
            conformal_radius=conformal_radius,
            tolerance=self._tolerance,
        )
        gate_results.append(gate2)
        explanations[gate2.name] = gate2.explanation
        metrics.update(gate2_metrics)

        verdict = Verdict.ACCEPT if (gate1.passed and gate2.passed) else Verdict.ABSTAIN

        report = GeneratedDecisionValidationReport(
            verdict=verdict,
            metrics=metrics,
            explanations=explanations,
            gate_results=tuple(gate_results),
        )
        outcome = ValidationOutcome(
            verdict=report.verdict,
            gate_results=report.gate_results,
        )
        return DecisionValidationCase(outcome=outcome, report=report)

    def _evaluate_ood_gate(
        self,
        *,
        candidate: NDArray[np.float64],
        ood_metrics: dict[str, float],
    ) -> tuple[GateResult, dict[str, float | bool]]:
        """Score the candidate against the Mahalanobis threshold."""
        try:
            md2 = float(ood_metrics["md2"])
            threshold_md2 = float(ood_metrics["threshold"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                "OOD calibrator must supply 'md2' and 'threshold' keys in transform()."
            ) from exc

        passed = md2 <= threshold_md2
        gate = GateResult(
            name="gate1",
            passed=bool(passed),
            metrics={"md2": md2, "threshold": threshold_md2},
            explanation=(
                "Pass: candidate is within the supported decision region."
                if passed
                else "ABSTAIN: candidate is outside the supported decision region."
            ),
        )
        metrics: dict[str, float | bool] = {
            "gate1_md2": md2,
            "gate1_md2_threshold": threshold_md2,
            "gate1_inlier": bool(passed),
        }
        return gate, metrics

    @staticmethod
    def _ensure_vector(array: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _apply_calibration(self, calibration: DecisionValidationCalibration) -> None:
        self._calibration = calibration
        self._ood_calibrator = calibration.ood_calibrator
        self._conformal_calibrator = calibration.conformal_calibrator

    # ------------------------------------------------------------------
    # Calibration metadata accessors
    # ------------------------------------------------------------------

    @property
    def calibration(self) -> DecisionValidationCalibration | None:
        return self._calibration

    @property
    def calibration_scope(self) -> str | None:
        return None if self._calibration is None else self._calibration.scope

    @property
    def calibration_threshold(self) -> float | None:
        if self._calibration is None:
            return None
        return self._calibration.ood_threshold

    @property
    def calibration_radius(self) -> float | None:
        if self._calibration is None:
            return None
        return self._calibration.conformal_radius

    def has_calibration(self) -> bool:
        return self._calibration is not None

    def _evaluate_tolerance_gate(
        self,
        *,
        X_target: NDArray[np.float64],
        X_hat: NDArray[np.float64],
        conformal_radius: float,
        tolerance: Tolerance,
    ) -> tuple[GateResult, dict[str, float | bool]]:
        """Compare predicted outcomes to target with conformal cushion."""
        diff = np.abs(X_hat - X_target)
        dist_l2 = float(np.linalg.norm(diff))
        q = float(conformal_radius)

        if tolerance.eps_l2 is not None:
            covered = (dist_l2 + q) <= float(tolerance.eps_l2)
        elif tolerance.eps_per_obj is not None:
            covered = bool(np.all(diff + q <= tolerance.eps_per_obj))
        else:  # pragma: no cover - enforced by Tolerance validators
            raise ValueError("Tolerance requires eps_l2 or eps_per_obj.")

        gate = GateResult(
            name="gate2",
            passed=bool(covered),
            metrics={"radius_q": q, "dist_to_target_l2": dist_l2},
            explanation=(
                "Pass: predicted X stays within tolerance after accounting for model error."
                if covered
                else "ABSTAIN: predicted X exceeds tolerance after accounting for model error."
            ),
        )
        metrics: dict[str, float | bool] = {
            "gate2_conformal_radius_q": q,
            "gate2_dist_to_target_l2": dist_l2,
            "gate2_covered": bool(covered),
        }
        return gate, metrics
