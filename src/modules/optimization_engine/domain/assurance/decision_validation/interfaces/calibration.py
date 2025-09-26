"""Calibration ports for decision validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....modeling.interfaces.base_estimator import BaseEstimator


@dataclass(frozen=True)
class ConformalTransformResult:
    prediction: NDArray[np.float64]
    radius: float


class OODCalibrator(ABC):
    @abstractmethod
    def fit(self, X_cal_norm: np.ndarray) -> None: ...

    @abstractmethod
    def transform(self, sample: np.ndarray) -> dict[str, float]: ...

    @property
    @abstractmethod
    def threshold(self) -> float: ...


class ConformalCalibrator(ABC):
    @abstractmethod
    def fit(
        self,
        X_cal_norm: np.ndarray,
        Y_cal_norm: np.ndarray,
        decision_estimator: BaseEstimator,
    ) -> None:
        """Fit calibration radius and (re)fit the provided estimator on normalized data."""

    @abstractmethod
    def transform(self, sample: np.ndarray) -> ConformalTransformResult: ...

    @property
    @abstractmethod
    def radius(self) -> float: ...
