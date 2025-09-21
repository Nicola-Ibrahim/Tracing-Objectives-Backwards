"""Forward model port producing mean predictions for validation."""

from abc import ABC, abstractmethod

import numpy as np


class ForwardModel(ABC):
    @abstractmethod
    def predict_mean(self, X: np.ndarray) -> np.ndarray: ...


__all__ = ["ForwardModel"]
