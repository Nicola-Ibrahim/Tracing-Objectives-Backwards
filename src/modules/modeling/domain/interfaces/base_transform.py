from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np


class TransformTarget(Enum):
    DECISIONS = "decisions"
    OBJECTIVES = "objectives"
    BOTH = "both"


class BaseTransformStep(ABC):
    """
    Abstract base class for preprocessing transform steps
    applied to data.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def target(self) -> TransformTarget:
        pass

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
    def from_fitted_state(
        cls, config: dict[str, Any], state: dict[str, Any]
    ) -> "BaseTransformStep":
        """Reconstruct the step from configuration and fitted state."""
        pass
