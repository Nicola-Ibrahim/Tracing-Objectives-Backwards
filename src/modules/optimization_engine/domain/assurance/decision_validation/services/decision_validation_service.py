from dataclasses import dataclass

import numpy as np

from ...feasibility.value_objects.tolerance import Tolerance
from ..aggregates import DecisionValidationCase
from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ..policies import evaluate_two_gate_policy
from ..strategies import (
    ConformalCalibration,
    ConformalSplitL2,
    ForwardEnsemble,
    OODCalibration,
    calibrate_mahalanobis,
)
from ..value_objects import ConfidenceLevel, OODCalibrationParams, ValidationOutcome


@dataclass
class DecisionValidationService:
    ensemble: ForwardEnsemble
    tolerance: Tolerance
    confidence: ConfidenceLevel
    ood_params: OODCalibrationParams

    _ood: OODCalibration | None = None
    _conformal: ConformalCalibration | None = None

    def calibrate(self, X_cal_norm: np.ndarray, Y_cal_norm: np.ndarray) -> None:
        self._ood = calibrate_mahalanobis(
            X_cal_norm,
            percentile=self.ood_params.percentile,
            cov_reg=self.ood_params.cov_reg,
        )
        conformal = ConformalSplitL2(self.confidence.value)
        conformal.fit(X_cal_norm, Y_cal_norm, self.ensemble)
        self._conformal = ConformalCalibration(radius_q=conformal.predictive_radius())

    @property
    def ood_calibration(self) -> OODCalibration:
        if self._ood is None:
            raise RuntimeError("Service not calibrated.")
        return self._ood

    @property
    def conformal_calibration(self) -> ConformalCalibration:
        if self._conformal is None:
            raise RuntimeError("Service not calibrated.")
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

        y_hat_norm = self.ensemble.predict_mean(np.atleast_2d(x_norm))[0]
        report: GeneratedDecisionValidationReport = evaluate_two_gate_policy(
            x_norm=x_norm,
            y_star_norm=y_star_norm,
            y_hat_norm=y_hat_norm,
            ood=self._ood,
            conf=self._conformal,
            tol=self.tolerance,
        )
        outcome = ValidationOutcome(verdict=report.verdict)
        return DecisionValidationCase(outcome=outcome, report=report)


__all__ = ["DecisionValidationService"]
