import numpy as np
from scipy.spatial.distance import cdist

from .base import BaseDiversityStrategy


class MaxMinDistanceDiversityStrategy(BaseDiversityStrategy):
    """
    Selects diverse points iteratively to maximize the minimum distance
    between selected points from the full Pareto front.
    """

    def select_diverse_points(
        self,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.shape[0] == 0 or num_suggestions == 0:
            return np.array([])

        # If num_suggestions is greater than or equal to available points, return all.
        if pareto_front_normalized.shape[0] <= num_suggestions:
            return pareto_front_normalized

        # Set seed for this selection if provided
        if self._random_seed is not None:
            np.random.seed(self._random_seed)

        selected_points = []

        # Start with the point closest to the target from the FULL front
        distances_to_target_from_full_front = np.linalg.norm(
            pareto_front_normalized - target_normalized, axis=1
        )
        first_idx = np.argmin(distances_to_target_from_full_front)
        selected_points.append(pareto_front_normalized[first_idx])

        # Indices of points not yet selected
        remaining_indices = np.delete(
            np.arange(pareto_front_normalized.shape[0]), first_idx
        )

        for _ in range(num_suggestions - 1):
            if len(remaining_indices) == 0:
                break

            # Calculate distances from remaining points to all currently selected points
            distances_to_selected = cdist(
                pareto_front_normalized[remaining_indices], np.array(selected_points)
            )
            # Find the minimum distance for each remaining point to any selected point
            min_distances_per_remaining = np.min(distances_to_selected, axis=1)

            # Select the point that has the maximum of these minimum distances
            next_idx_in_remaining = np.argmax(min_distances_per_remaining)
            next_point_original_idx = remaining_indices[next_idx_in_remaining]

            selected_points.append(pareto_front_normalized[next_point_original_idx])
            remaining_indices = np.delete(remaining_indices, next_idx_in_remaining)

        return np.array(selected_points)
