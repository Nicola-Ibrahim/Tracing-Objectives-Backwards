import numpy as np

from .....domain.assurance.decision_validation.interfaces import (
    ConformalCalibrator,
    ConformalTransformResult,
)
from .....domain.modeling.interfaces.base_estimator import BaseEstimator


class SplitConformalL2Calibrator(ConformalCalibrator):
    """Split conformal calibration using L2 norm."""

    def __init__(self, confidence: float = 0.90) -> None:
        """Initialize the calibrator with the desired confidence level.
        Args:
            confidence (float, optional): Confidence level for the conformal set.
                Must be in (0,1). Defaults to 0.90.
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0,1)")
        self._confidence = confidence
        self._radius: float | None = None
        self._estimator: BaseEstimator | None = None

    def fit(
        self,
        X_cal_norm: np.ndarray,
        Y_cal_norm: np.ndarray,
        estimator: BaseEstimator,
    ) -> None:
        """Fit estimator on normalized data before computing conformal radius."""
        X_cal_norm = np.asarray(X_cal_norm, dtype=float)
        Y_cal_norm = np.asarray(Y_cal_norm, dtype=float)

        estimator.fit(X_cal_norm, Y_cal_norm)
        y_hat = estimator.predict(X_cal_norm)
        residuals = np.linalg.norm(Y_cal_norm - y_hat, axis=1)
        self._radius = float(np.quantile(residuals, self._confidence))
        self._estimator = estimator

    def transform(self, sample: np.ndarray) -> ConformalTransformResult:
        if self._radius is None or self._estimator is None:
            raise RuntimeError(
                "SplitConformalL2Calibrator must be fitted before transform()."
            )

        sample = np.asarray(sample, dtype=float)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        prediction = self._estimator.predict(sample)
        prediction = np.asarray(prediction, dtype=float)
        return ConformalTransformResult(prediction=prediction, radius=self._radius)

    @property
    def radius(self) -> float:
        if self._radius is None:
            raise RuntimeError("SplitConformalL2Calibrator has not been fitted yet.")
        return self._radius
