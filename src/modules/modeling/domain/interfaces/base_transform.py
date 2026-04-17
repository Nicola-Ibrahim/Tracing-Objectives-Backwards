from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTransformer(ABC):
    """
    Abstract base class for preprocessing transform steps
    applied to data.
    """

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Serializable configuration for the transform step."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_fitted_state(self) -> dict[str, Any]:
        """Return variables required to restore the fitted state."""
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "BaseTransformer":
        """Reconstruct the step from configuration and fitted state dict."""
        pass
