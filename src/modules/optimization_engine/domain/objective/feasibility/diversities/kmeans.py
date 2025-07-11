import warnings

import numpy as np
from sklearn.cluster import KMeans

from .base import BaseDiversityStrategy


class KMeansDiversityStrategy(BaseDiversityStrategy):
    """
    Uses K-Means clustering to find `num_desired_points` representative points
    from the pool. Returns the points from the pool closest to the cluster centroids.
    """

    def select_diverse_points(
        self,
        pool_points: np.ndarray,
        num_desired_points: int,
        target_normalized: np.ndarray,  # Not directly used for selection, but part of interface
    ) -> np.ndarray:
        if pool_points.shape[0] == 0:
            return np.array([])
        if num_desired_points == 0:
            return np.array([])

        if pool_points.shape[0] < num_desired_points:
            warnings.warn(
                f"Not enough points in the pool ({pool_points.shape[0]}) for K-Means with {num_desired_points} clusters. "
                "Returning all available points."
            )
            return pool_points

        # Handle num_desired_points == 1 edge case for KMeans
        if num_desired_points == 1:
            # If only one point is needed, return the one closest to the target
            distances_to_target = np.linalg.norm(
                pool_points - target_normalized, axis=1
            )
            return pool_points[np.argmin(distances_to_target)].reshape(1, -1)

        kmeans = KMeans(
            n_clusters=num_desired_points,
            random_state=self._random_seed,
            n_init="auto",
        )
        kmeans.fit(pool_points)

        selected_points = []
        cluster_centroids = kmeans.cluster_centers_

        # For each centroid, find the actual point in the pool that is closest to it
        for centroid in cluster_centroids:
            distances_to_centroid = np.linalg.norm(pool_points - centroid, axis=1)
            closest_point_idx = np.argmin(distances_to_centroid)
            selected_points.append(pool_points[closest_point_idx])

        # Ensure unique points are returned if multiple centroids map to the same point
        return np.unique(np.array(selected_points), axis=0)
