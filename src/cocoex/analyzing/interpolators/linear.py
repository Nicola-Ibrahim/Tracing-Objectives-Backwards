import numpy as np
from numpy.typing import NDArray

from ..preference import ObjectivePreferences
from ..similarities import SimilarityMetric
from .base import BaseInterpolator


class LinearInterpolator(BaseInterpolator):
    """
    Linear interpolator that recommends a decision vector based on preference-weighted similarity to objective vectors.
    Assumes parameterized data and uses linear interpolation along the front.
    """

    def __init__(self, similarity_metric: SimilarityMetric):
        self.similarity_metric = similarity_metric
        self.param_coords: NDArray[np.float64] = np.array([])
        self.decision_vectors: NDArray[np.float64] = np.array([])
        self.objective_vectors: NDArray[np.float64] = np.array([])

    def fit(
        self,
        decision_vectors: NDArray[np.float64],
        objective_vectors: NDArray[np.float64],
    ) -> None:
        if len(decision_vectors) != len(objective_vectors):
            raise ValueError(
                "Decision and objective vectors must have the same length."
            )

        # Sort by the first objective to establish a consistent order
        sorted_idx = np.argsort(objective_vectors[:, 0])
        self.decision_vectors = decision_vectors[sorted_idx]
        self.objective_vectors = objective_vectors[sorted_idx]

        if len(decision_vectors) == 1:
            self.param_coords = np.array([0.5])
        else:
            self.param_coords = np.linspace(0.0, 1.0, len(decision_vectors))

    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        if len(self.param_coords) == 0:
            raise ValueError("Interpolator has not been fitted.")

        weights = np.array([preferences.time_weight, preferences.energy_weight])
        similarity_scores = self.similarity_metric(self.objective_vectors, weights)
        total_similarity = np.sum(similarity_scores)

        if abs(total_similarity) < 1e-12:
            alpha = float(np.mean(self.param_coords))
        else:
            alpha = float(
                np.dot(self.param_coords, similarity_scores) / total_similarity
            )

        return self._interpolate(alpha)

    def _interpolate(self, query_coord: float) -> NDArray[np.float64]:
        if len(self.param_coords) == 1:
            return self.decision_vectors[0]

        idx = np.searchsorted(self.param_coords, query_coord)

        if idx == 0:
            return self.decision_vectors[0]
        if idx == len(self.param_coords):
            return self.decision_vectors[-1]

        s0, s1 = self.param_coords[idx - 1], self.param_coords[idx]
        v0, v1 = self.decision_vectors[idx - 1], self.decision_vectors[idx]

        if s1 <= s0:
            return v0 if query_coord - s0 < s1 - query_coord else v1

        t = (query_coord - s0) / (s1 - s0)
        t = np.clip(t, 0.0, 1.0)

        return (1 - t) * v0 + t * v1
