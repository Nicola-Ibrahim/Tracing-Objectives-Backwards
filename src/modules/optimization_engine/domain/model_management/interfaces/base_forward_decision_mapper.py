from abc import ABC, abstractmethod

import numpy as np


class BaseForwardDecisionMapper(ABC):
    """
    Abstract base class for a forward model, mapping a design/structure (decision)
    to its predicted performance/objective (objective).
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract method to predict targets from input features.

        Args:
            X: Input features (numpy array).

        Returns:
            Predicted targets (numpy array).
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Abstract method for training the forward mapper.
        This method will typically be implemented for learnable mappers (e.g., neural networks).

        Args:
            X: Training features (numpy array).
            y: Training targets (numpy array).
        """
        pass
