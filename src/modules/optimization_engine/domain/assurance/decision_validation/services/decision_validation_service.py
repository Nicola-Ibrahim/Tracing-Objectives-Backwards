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
    """Run OOD and feasibility gates for a normalized y decision."""

    def __init__(
        self,
        *,
        tolerance: float,
        ood_calibrator: BaseOODCalibrator,
        conformal_calibrator: BaseConformalCalibrator,
    ) -> None:
        if tolerance is None:
            raise ValueError(
                "Provide tolerance or eps_per_obj for decision validation."
            )

        self._tolerance = self._coerce_tolerance(tolerance)
        self._ood_calibrator = ood_calibrator
        self._conformal_calibrator = conformal_calibrator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        y: NDArray[np.float64] | None = None,
        X_target: NDArray[np.float64] | None = None,
        **aliases: NDArray[np.float64],
    ) -> GeneratedDecisionValidationReport:
        """Evaluate both assurance gates for a y decision."""
        y_arr, X_target_arr = self._resolve_aliases(y, X_target, aliases)

        y = self._as_vector(y_arr, "y")
        X_target = self._as_vector(X_target_arr, "X_target")

        raw_ood_result = self._ood_calibrator.evaluate(y)
        raw_conformal_result = self._conformal_calibrator.evaluate(
            y=y,
            X_target=X_target,
            tolerance=self._tolerance,
        )

        gate1 = self._raw_to_gate_result("gate1", raw_ood_result)
        gate2 = self._raw_to_gate_result("gate2", raw_conformal_result)
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
        y: NDArray[np.float64] | None,
        X_target: NDArray[np.float64] | None,
        aliases: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if y is None:
            y = aliases.pop("y_norm", None)
        if X_target is None:
            X_target = aliases.pop("X_target_norm", None)
        if aliases:
            unexpected = ", ".join(sorted(aliases))
            raise TypeError(f"Unexpected keyword(s) for validate(): {unexpected}")
        if y is None or X_target is None:
            raise TypeError(
                "validate() requires both 'y' and 'X_target' arrays (or their aliases)."
            )
        return y, X_target

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
