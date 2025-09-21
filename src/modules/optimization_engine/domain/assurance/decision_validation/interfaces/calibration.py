"""Ports for calibration strategies used during decision validation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..value_objects import ConformalCalibration, OODCalibration

if TYPE_CHECKING:  # pragma: no cover
    from .forward_model import ForwardModel


class OODCalibrator(ABC):
    @abstractmethod
    def fit(self, X_cal_norm: np.ndarray) -> OODCalibration: ...


class ConformalCalibrator(ABC):
    @abstractmethod
    def fit(
        self,
        X_cal_norm: np.ndarray,
        Y_cal_norm: np.ndarray,
        forward_model: "ForwardModel",
    ) -> ConformalCalibration: ...


__all__ = ["OODCalibrator", "ConformalCalibrator"]
