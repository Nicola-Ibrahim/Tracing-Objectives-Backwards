import warnings

import numpy as np
from sklearn.cluster import KMeans

from .base import BaseDiversityStrategy


class KMeansDiversityStrategy(BaseDiversityStrategy):
    """
    Uses K-Means clustering to find `num_suggestions` representative points
    from the pool. Returns the points from the pool closest to the cluster centroids.
    """

    def select_diverse_points(
        self,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.shape[0] == 0 or num_suggestions == 0:
            return np.array([])

        if pareto_front_normalized.shape[0] < num_suggestions:
            warnings.warn(
                f"Not enough points in the Pareto front ({pareto_front_normalized.shape[0]}) for K-Means with {num_suggestions} clusters. "
                "Returning all available points."
            )
            return pareto_front_normalized

        # Handle num_suggestions == 1 edge case for KMeans
        if num_suggestions == 1:
            # If only one point is needed, return the one closest to the target from the full front
            distances_to_target = np.linalg.norm(
                pareto_front_normalized - target_normalized, axis=1
            )
            return pareto_front_normalized[np.argmin(distances_to_target)].reshape(
                1, -1
            )

        # Set seed for KMeans for reproducibility
        kmeans = KMeans(
            n_clusters=num_suggestions,
            random_state=self._random_seed,
            n_init="auto",
        )
        kmeans.fit(pareto_front_normalized)  # Fit on the full front

        selected_points = []
        cluster_centroids = kmeans.cluster_centers_

        # For each centroid, find the actual point in the full front that is closest to it
        for centroid in cluster_centroids:
            distances_to_centroid = np.linalg.norm(
                pareto_front_normalized - centroid, axis=1
            )
            closest_point_idx = np.argmin(distances_to_centroid)
            selected_points.append(pareto_front_normalized[closest_point_idx])

        # Ensure unique points are returned if multiple centroids map to the same point
        return np.unique(np.array(selected_points), axis=0)
