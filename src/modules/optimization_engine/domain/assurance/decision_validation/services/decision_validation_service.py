from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ..enums.verdict import Verdict
from ..interfaces.base_conformal_calibrator import BaseConformalCalibrator
from ..interfaces.base_ood_calibrator import BaseOODCalibrator
from ..value_objects.gate_result import GateResult


class DecisionValidationService:
    """Run OOD and feasibility gates for a normalized candidate decision."""

    def __init__(
        self,
        *,
        eps_l2: float | None = None,
        eps_per_obj: Sequence[float] | None = None,
        ood_calibrator: BaseOODCalibrator,
        conformal_calibrator: BaseConformalCalibrator,
    ) -> None:
        if eps_l2 is None and eps_per_obj is None:
            raise ValueError("Provide eps_l2 or eps_per_obj for decision validation.")

        self._eps_l2 = self._coerce_eps_l2(eps_l2)
        self._eps_per_obj = self._coerce_eps_per_obj(eps_per_obj)
        self._ood_calibrator = ood_calibrator
        self._conformal_calibrator = conformal_calibrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        y_candidate: NDArray[np.float64] | None = None,
        X_target: NDArray[np.float64] | None = None,
        **aliases: NDArray[np.float64],
    ) -> GeneratedDecisionValidationReport:
        """Evaluate both assurance gates for a candidate decision."""
        candidate_arr, target_arr = self._resolve_aliases(
            y_candidate, X_target, aliases
        )

        candidate_vec = self._as_vector(candidate_arr, "candidate")
        target_vec = self._as_vector(target_arr, "target")

        raw_gate1 = self._ood_calibrator.evaluate(candidate_vec)
        raw_gate2 = self._conformal_calibrator.evaluate(
            candidate=candidate_vec,
            target=target_vec,
            eps_l2=self._eps_l2,
            eps_per_obj=self._eps_per_obj,
        )

        gate1 = self._raw_to_gate_result("gate1", raw_gate1)
        gate2 = self._raw_to_gate_result("gate2", raw_gate2)
        gate_results = (gate1, gate2)

        metrics: dict[str, float | bool] = {}
        for gate in gate_results:
            for key, value in gate.metrics.items():
                metrics[f"{gate.name}_{key}"] = value

        verdict = (
            Verdict.ACCEPT
            if all(gate.passed == Verdict.ACCEPT for gate in gate_results)
            else Verdict.ABSTAIN
        )

        return GeneratedDecisionValidationReport(
            verdict=verdict,
            metrics=metrics,
            gate_results=gate_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_aliases(
        candidate: NDArray[np.float64] | None,
        target: NDArray[np.float64] | None,
        aliases: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        return candidate, target

    @staticmethod
    def _as_vector(array: NDArray[np.float64], label: str) -> NDArray[np.float64]:
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim != 1:
            raise ValueError(
                f"Expected '{label}' to be a 1-D vector or a single-sample 2-D array."
            )
        return arr

    @staticmethod
    def _coerce_eps_l2(value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("eps_l2 must be non-negative.")
        return float(value)

    @staticmethod
    def _coerce_eps_per_obj(
        values: Sequence[float] | None,
    ) -> NDArray[np.float64] | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=float)
        if np.any(arr < 0):
            raise ValueError("eps_per_obj entries must be non-negative.")
        return arr

    @staticmethod
    def _raw_to_gate_result(
        name: str, raw_result: tuple[bool, dict[str, float | bool], str]
    ) -> GateResult:
        passed_flag, metrics, explanation = raw_result
        verdict = Verdict.ACCEPT if passed_flag else Verdict.ABSTAIN
        return GateResult(
            name=name,
            passed=verdict,
            metrics=dict(metrics),
            explanation=explanation,
        )
