"""Infrastructure diversity strategy using scikit-learn KMeans."""

import warnings

import numpy as np
from sklearn.cluster import KMeans

from ....domain.assurance.feasibility.interfaces.diversity import BaseDiversityStrategy


class KMeansDiversityStrategy(BaseDiversityStrategy):
    def __init__(self, random_seed: int = 42, n_clusters: int = 5):
        self._random_seed = random_seed
        self._n_clusters = n_clusters

    def select_diverse_points(
        self,
        *,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.size == 0 or num_suggestions <= 0:
            return np.empty((0, pareto_front_normalized.shape[1]))

        if pareto_front_normalized.shape[0] < num_suggestions:
            warnings.warn(
                "Insufficient Pareto points for KMeans diversity; returning all points."
            )
            return pareto_front_normalized

        if num_suggestions == 1:
            distances = np.linalg.norm(
                pareto_front_normalized - target_normalized, axis=1
            )
            return pareto_front_normalized[np.argmin(distances)].reshape(1, -1)

        kmeans = KMeans(
            n_clusters=num_suggestions,
            random_state=self._random_seed,
            n_init="auto",
        )
        kmeans.fit(pareto_front_normalized)

        selected: list[np.ndarray] = []
        for centroid in kmeans.cluster_centers_:
            distances = np.linalg.norm(pareto_front_normalized - centroid, axis=1)
            selected.append(pareto_front_normalized[np.argmin(distances)])

        return np.unique(np.vstack(selected), axis=0)
