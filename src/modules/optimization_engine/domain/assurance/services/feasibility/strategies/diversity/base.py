from abc import ABC, abstractmethod

import numpy as np


class BaseDiversityStrategy(ABC):
    """
    Abstract Base Class for strategies that select diverse points from a pool.
    """

    def __init__(self, random_seed: int = None):
        self._random_seed = random_seed

    @abstractmethod
    def select_diverse_points(
        self,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        """
        Selects a diverse subset of points from the given full Pareto front.

        Args:
            pareto_front_normalized (np.ndarray): The entire historical Pareto front
                                                 in normalized space. Expected shape (N, D).
            target_normalized (np.ndarray): The target point (normalized) that the suggestions
                                            are trying to move towards (useful for initial point selection).
            num_suggestions (int): The number of diverse points to select.

        Returns:
            np.ndarray: A 2D array of selected diverse points.
        """
        pass
