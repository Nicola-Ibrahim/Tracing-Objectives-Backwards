from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ....modeling.interfaces.base_estimator import BaseEstimator


class BaseConformalCalibrator(ABC):
    """Base contract for conformal calibrators coupled with an estimator."""

    def __init__(self, estimator: BaseEstimator) -> None:
        if estimator is None:
            raise ValueError("BaseConformalCalibrator requires an estimator instance.")
        self._estimator = estimator

    @property
    def estimator(self) -> BaseEstimator:
        """Return the estimator trained alongside the calibrator."""
        return self._estimator

    def fit(
        self,
        Y_cal_norm: np.ndarray,
        X_cal_norm: np.ndarray,
    ) -> None:
        """
        Fit the attached estimator and calibrator-specific components on normalized data.

        Subclasses should implement ``_fit_calibrator`` to compute additional
        calibration statistics (e.g., conformal radii) using the provided arrays.
        """

        Y_cal = np.asarray(Y_cal_norm, dtype=float)
        X_cal = np.asarray(X_cal_norm, dtype=float)
        self._estimator.fit(Y_cal, X_cal)
        self._fit_calibrator(Y_cal, X_cal)

    @abstractmethod
    def _fit_calibrator(
        self,
        Y_cal_norm: np.ndarray,
        X_cal_norm: np.ndarray,
    ) -> None:
        """Subclass-specific calibration logic executed after estimator fitting."""

    @abstractmethod
    def transform(self, sample: np.ndarray) -> Any: ...

    @abstractmethod
    def evaluate(
        self,
        *,
        candidate: np.ndarray,
        target: np.ndarray,
        eps_l2: float | None,
        eps_per_obj: Sequence[float] | np.ndarray | None,
    ) -> tuple[bool, dict[str, float | bool], str]: ...

    @property
    @abstractmethod
    def radius(self) -> float: ...
