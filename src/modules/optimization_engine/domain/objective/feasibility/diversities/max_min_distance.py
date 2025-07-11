import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseDiversityStrategy


class MaxMinDistanceDiversityStrategy(BaseDiversityStrategy):
    """
    Selects diverse points iteratively to maximize the minimum distance
    between selected points.
    """

    def select_diverse_points(
        self,
        pool_points: np.ndarray,
        num_desired_points: int,
        target_normalized: np.ndarray,
    ) -> np.ndarray:
        if pool_points.shape[0] == 0:
            return np.array([])
        if num_desired_points == 0:
            return np.array([])
        if pool_points.shape[0] <= num_desired_points:
            return pool_points

        # Set seed for this selection if provided
        if self._random_seed is not None:
            np.random.seed(self._random_seed)

        selected_points = []

        # Start with the point closest to the target from the pool
        distances_to_target = np.linalg.norm(pool_points - target_normalized, axis=1)
        first_idx_in_pool = np.argmin(distances_to_target)
        selected_points.append(pool_points[first_idx_in_pool])

        remaining_indices = np.delete(
            np.arange(pool_points.shape[0]), first_idx_in_pool
        )

        for _ in range(num_desired_points - 1):
            if len(remaining_indices) == 0:
                break

            # Calculate distances from remaining points to all currently selected points
            distances_to_selected = cdist(
                pool_points[remaining_indices], np.array(selected_points)
            )
            # Find the minimum distance for each remaining point to any selected point
            min_distances_per_remaining = np.min(distances_to_selected, axis=1)

            # Select the point that has the maximum of these minimum distances
            next_idx_in_remaining = np.argmax(min_distances_per_remaining)
            next_point_original_idx = remaining_indices[next_idx_in_remaining]

            selected_points.append(pool_points[next_point_original_idx])
            remaining_indices = np.delete(remaining_indices, next_idx_in_remaining)

        return np.array(selected_points)
