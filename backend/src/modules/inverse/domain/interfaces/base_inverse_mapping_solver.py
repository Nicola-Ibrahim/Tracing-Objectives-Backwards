from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class InverseSolverResult(BaseModel):
    """Value object representing the complete result of a generation cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates_X: np.ndarray
    candidates_y: np.ndarray
    metadata: dict[str, Any]


class AbstractInverseMappingSolver(ABC):
    """
    The polymorphic interface for all inverse mapping strategies.
    """

    @abstractmethod
    def type(self) -> str:
        """Returns the type of the solver."""
        raise NotImplementedError("Solver type not implemented.")

    @abstractmethod
    def history(self) -> dict[str, Any]:
        """Returns the history of the solver."""
        raise NotImplementedError("History not implemented.")

    @abstractmethod
    def _ensure_fitted(self) -> None:
        """Raises RuntimeError if the solver is not fitted."""
        raise RuntimeError("Solver not fitted. Call 'train' first.")

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Trains the solver."""
        raise NotImplementedError("Solver training not implemented.")

    @abstractmethod
    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        """Executes the specific inverse mapping logic."""
        raise NotImplementedError("Solver generation not implemented.")
