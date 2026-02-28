from abc import ABC, abstractmethod

import numpy as np


class BaseSamplingStrategy(ABC):
    """
    Standard interface for all candidate sampling algorithms within the domain.
    Enables the Strategy Pattern for interchangeable sampling logic.
    """

    @abstractmethod
    def sample(
        self,
        anchors: np.ndarray,
        sampling_params: dict,
    ) -> np.ndarray:
        """
        Mathematical sampling operation in normalized space.

        Args:
            anchors: The normalized decision-space configurations of the anchors.
            sampling_params: Dictionary of strategy-specific parameters.

        Returns:
            (n_samples, D) array of normalized decision space candidates.
        """
        raise NotImplementedError("Subclasses must implement this method")
