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
        pool_points: np.ndarray,
        num_desired_points: int,
        # target_normalized is passed here if strategies might need it for their logic
        target_normalized: np.ndarray,  # This might be useful for some selection methods (e.g., initial point)
    ) -> np.ndarray:
        """
        Selects a diverse subset of points from the given pool.

        Args:
            pool_points (np.ndarray): The pool of points (e.g., nearest Pareto points)
                                      from which to select. Expected shape (N, D).
            num_desired_points (int): The number of diverse points to select.
            target_normalized (np.ndarray): The target point (normalized) that the suggestions
                                            are trying to move towards (useful for initial point selection).

        Returns:
            np.ndarray: A 2D array of selected diverse points.
        """
        pass
