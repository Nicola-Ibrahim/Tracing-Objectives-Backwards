from abc import ABC, abstractmethod

import numpy as np


class BaseForwardDecisionMapper(ABC):
    """
    Abstract base class for a forward model, mapping a design/structure (decision)
    to its predicted performance/objective (objective).
    """

    @abstractmethod
    def predict(self, target_decision: np.ndarray) -> np.ndarray:
        """
        Abstract method to predict the performance (objectives) from a given target decision.

        Args:
            target_decision: The input decision (numpy array).

        Returns:
            The predicted objectives (numpy array).
        """
        pass

    @abstractmethod
    def fit(self, decisions: np.ndarray, objectives: np.ndarray):
        """
        Abstract method for training the forward mapper.
        This method will typically be implemented only for learnable mappers (e.g., neural networks).

        Args:
            decisions: Training data for decisions (numpy array).
            objectives: Training data for objectives (numpy array).
        """
        pass
