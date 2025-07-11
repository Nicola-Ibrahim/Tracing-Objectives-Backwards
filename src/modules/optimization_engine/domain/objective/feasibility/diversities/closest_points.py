import numpy as np

from .base import BaseDiversityStrategy


class ClosestPointsDiversityStrategy(BaseDiversityStrategy):
    """
    Simply returns the `num_desired_points` closest points from the pool
    to the target. This is the "none" or default diversity method.
    """

    def select_diverse_points(
        self,
        pool_points: np.ndarray,
        num_desired_points: int,
        target_normalized: np.ndarray,  # Used for ordering points by closeness
    ) -> np.ndarray:
        if pool_points.shape[0] == 0:
            return np.array([])
        if num_desired_points == 0:
            return np.array([])

        # Points are already sorted by distance to target when pool is created,
        # so we just take the first `num_desired_points`.
        return pool_points[:num_desired_points]
