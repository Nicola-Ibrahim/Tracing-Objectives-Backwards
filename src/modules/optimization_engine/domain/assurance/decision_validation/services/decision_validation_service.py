"""Decision validation service orchestrating policies via injected ports."""

from typing import Optional

import numpy as np

from ...feasibility.value_objects.tolerance import Tolerance
from ..aggregates import DecisionValidationCase
from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ..interfaces import (
    ConformalCalibrator,
    ForwardModel,
    OODCalibrator,
)
from ..policies import evaluate_two_gate_policy
from ..value_objects import (
    ConformalCalibration,
    OODCalibration,
    ValidationOutcome,
)


class DecisionValidationService:
    def __init__(
        self,
        *,
        forward_model: ForwardModel,
        tolerance: Tolerance,
        ood_calibrator: OODCalibrator,
        conformal_calibrator: ConformalCalibrator,
    ) -> None:
        self._forward_model = forward_model
        self._tolerance = tolerance
        self._ood_calibrator = ood_calibrator
        self._conformal_calibrator = conformal_calibrator
        self._ood: Optional[OODCalibration] = None
        self._conformal: Optional[ConformalCalibration] = None

    def calibrate(self, X_cal_norm: np.ndarray, Y_cal_norm: np.ndarray) -> None:
        self._ood = self._ood_calibrator.fit(X_cal_norm)
        self._conformal = self._conformal_calibrator.fit(
            X_cal_norm,
            Y_cal_norm,
            self._forward_model,
        )

    @property
    def ood_calibration(self) -> OODCalibration:
        if self._ood is None:
            raise RuntimeError(
                "DecisionValidationService must be calibrated before use."
            )
        return self._ood

    @property
    def conformal_calibration(self) -> ConformalCalibration:
        if self._conformal is None:
            raise RuntimeError(
                "DecisionValidationService must be calibrated before use."
            )
        return self._conformal

    def validate(
        self,
        *,
        x_norm: np.ndarray,
        y_star_norm: np.ndarray,
    ) -> DecisionValidationCase:
        if self._ood is None or self._conformal is None:
            raise RuntimeError(
                "DecisionValidationService must be calibrated before validation."
            )

        y_hat_norm = self._forward_model.predict_mean(np.atleast_2d(x_norm))[0]
        report: GeneratedDecisionValidationReport = evaluate_two_gate_policy(
            x_norm=x_norm,
            y_star_norm=y_star_norm,
            y_hat_norm=y_hat_norm,
            ood=self._ood,
            conf=self._conformal,
            tol=self._tolerance,
        )
        outcome = ValidationOutcome(
            verdict=report.verdict,
            gate_results=report.gate_results,
        )
        return DecisionValidationCase(outcome=outcome, report=report)


__all__ = ["DecisionValidationService"]
