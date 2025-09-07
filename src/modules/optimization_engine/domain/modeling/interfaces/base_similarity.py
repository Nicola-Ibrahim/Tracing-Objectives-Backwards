from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Define the type alias within the interface or domain for clarity
SimilarityMethod = Callable[
    [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
]


class BaseSimilarityCalculator(ABC):
    """
    Abstract Base Class for similarity calculation strategies.
    Defines the interface for any similarity calculation method.
    """

    @abstractmethod
    def calculate_similarity(
        self,
        array: NDArray[np.floating],
        vector: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Calculates the similarity between each row of 'array' and a 1D 'vector'.

        Args:
            array: A 2D NumPy array with shape (n_samples, n_features).
            vector: A 1D NumPy array with shape (n_features,).

        Returns:
            A 1D NumPy array of similarities with shape (n_samples,).
        """
        pass
