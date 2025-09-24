"""Infrastructure conformal calibration using split L2 coverage."""

import numpy as np

from .....domain.assurance.decision_validation.interfaces import (
    ConformalCalibrator,
    ForwardModel,
)
from .....domain.assurance.decision_validation.value_objects.calibration import (
    ConformalCalibration,
)


class SplitConformalL2Calibrator(ConformalCalibrator):
    def __init__(self, confidence: float = 0.90) -> None:
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0,1)")
        self._confidence = confidence

    def fit(
        self,
        X_cal_norm: np.ndarray,
        Y_cal_norm: np.ndarray,
        forward_model: ForwardModel,
    ) -> ConformalCalibration:
        y_hat = forward_model.predict_mean(X_cal_norm)
        residuals = np.linalg.norm(Y_cal_norm - y_hat, axis=1)
        radius = float(np.quantile(residuals, self._confidence))
        return ConformalCalibration(radius_q=radius)


__all__ = ["SplitConformalL2Calibrator"]
