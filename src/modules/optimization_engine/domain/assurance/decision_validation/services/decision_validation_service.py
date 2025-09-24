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
from ..value_objects.calibration import ConformalCalibration, OODCalibration
from ..value_objects.validation_outcome import ValidationOutcome


class DecisionValidationService:
    """Applies decision-validation gates using injected calibration ports."""

    def __init__(
        self,
        *,
        tolerance: Tolerance,
        ood_calibrator: OODCalibrator,
        conformal_calibrator: Optional[ConformalCalibrator] = None,
        forward_model: Optional[ForwardModel] = None,
    ) -> None:
        """Persist validation ports and enforce forward model requirements."""
        if conformal_calibrator is not None and forward_model is None:
            raise ValueError(
                "Forward model is required when a conformal calibrator is supplied."
            )
        self._tolerance = tolerance
        self._ood_calibrator = ood_calibrator
        self._conformal_calibrator = conformal_calibrator
        self._forward_model = forward_model

        self._ood: Optional[OODCalibration] = None
        self._conformal: Optional[ConformalCalibration] = None

    def calibrate(self, y_cal_norm: np.ndarray, x_cal_norm: np.ndarray) -> None:
        """Calibrate detectors using normalized decision and objective data."""

        self._ood = self._ood_calibrator.fit(y_cal_norm)

        if self._conformal_calibrator is not None and self._forward_model is not None:
            self._conformal = self._conformal_calibrator.fit(
                y_cal_norm,
                x_cal_norm,
                self._forward_model,
            )
        else:
            self._conformal = None

    @property
    def ood_calibration(self) -> OODCalibration:
        """Return the fitted OOD calibration; requires prior calibration."""
        if self._ood is None:
            raise RuntimeError(
                "DecisionValidationService must be calibrated before use."
            )
        return self._ood

    @property
    def conformal_calibration(self) -> Optional[ConformalCalibration]:
        """Return the fitted conformal calibration if available."""
        return self._conformal

    def validate(
        self,
        *,
        y_norm: np.ndarray,
        x_target_norm: np.ndarray,
    ) -> DecisionValidationCase:
        """Evaluate y against two assurance gates and return the decision case."""
        if self._ood is None:
            raise RuntimeError(
                "DecisionValidationService must be calibrated before validation."
            )

        prediction = self.predict_x(y_norm)
        if prediction is not None:
            x_hat_norm = prediction[0]
            conf = self._conformal or ConformalCalibration(radius_q=float("inf"))
        else:
            # Without a predictive model we can only assert inlier status.
            x_hat_norm = x_target_norm
            conf = ConformalCalibration(radius_q=float("inf"))

        report: GeneratedDecisionValidationReport = evaluate_two_gate_policy(
            y_norm=y_norm,
            x_target_norm=x_target_norm,
            x_hat_norm=x_hat_norm,
            ood=self._ood,
            conf=conf,
            tol=self._tolerance,
        )

        outcome = ValidationOutcome(
            verdict=report.verdict,
            gate_results=report.gate_results,
        )
        return DecisionValidationCase(outcome=outcome, report=report)

    def has_forward_model(self) -> bool:
        """Indicate whether a forward model is available for predictive checks."""
        return self._forward_model is not None

    def predict_x(self, y_norm: np.ndarray) -> Optional[np.ndarray]:
        """Predict mean x from normalized y if a forward model exists."""
        if self._forward_model is None:
            return None
        return self._forward_model.predict_mean(np.atleast_2d(y_norm))


__all__ = ["DecisionValidationService"]
