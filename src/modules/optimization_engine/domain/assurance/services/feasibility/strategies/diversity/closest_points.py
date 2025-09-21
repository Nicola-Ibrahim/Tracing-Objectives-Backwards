import numpy as np

from .base import BaseDiversityStrategy


class ClosestPointsDiversityStrategy(BaseDiversityStrategy):
    """
    Returns the `num_suggestions` closest points from the full Pareto front
    to the target. This acts as the "none" or default diversity method.
    """

    def select_diverse_points(
        self,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.shape[0] == 0 or num_suggestions == 0:
            return np.array([])

        # Calculate distances to target from the FULL front
        distances = np.linalg.norm(pareto_front_normalized - target_normalized, axis=1)
        # Get indices of the closest points
        closest_indices = np.argsort(distances)[:num_suggestions]
        # Return the actual closest points
        return pareto_front_normalized[closest_indices]
