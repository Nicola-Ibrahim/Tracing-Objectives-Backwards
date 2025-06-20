from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ....domain import ObjectivePreferences
from ....domain import BaseInterpolator
from ...similarities import SimilarityMethod


class KNearestNeighborInterpolator(BaseInterpolator):
    """
    K-Nearest Neighbors interpolator that selects top K candidate solutions closest
    to the user preference in the objective space and returns a weighted average
    of their decision vectors.
    """

    def __init__(self, similarity_metric: SimilarityMethod, k: int = 3) -> None:
        """
        Initialize the KNearestNeighborInterpolator.

        Args:
            similarity_metric (SimilarityMethod): Similarity metric used to compare objective vectors.
            k (int): Number of nearest neighbors to use for interpolation.
        """
        self.similarity_metric = similarity_metric
        self.k = k
        self._candidate_solutions: NDArray[np.float64] | None = None
        self._objective_front: NDArray[np.float64] | None = None
        self._active_indices: Sequence[int] | None = None

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fit the interpolator with candidate solutions and corresponding objective vectors.

        Args:
            candidate_solutions (NDArray[np.float64]): Array of candidate solutions in decision space, shape (N, D).
            objective_front (NDArray[np.float64]): Array of objective vectors, shape (N, M).

        Raises:
            ValueError: If lengths of candidate_solutions and objective_front do not match.
        """
        if len(candidate_solutions) != len(objective_front):
            raise ValueError(
                "Candidate solutions and objective front must be the same length."
            )

        self._candidate_solutions = candidate_solutions
        self._objective_front = objective_front
        self._active_indices = list(range(len(candidate_solutions)))

    def select_active_subset(self, subset_indices: Sequence[int] | None = None) -> None:
        """
        Select a subset of candidate solutions to be considered active for recommendations.

        Args:
            subset_indices (Sequence[int] | None): Indices of candidate solutions to activate.
                If None, activates all candidate solutions.
        """
        if subset_indices is None:
            self._active_indices = list(range(len(self._candidate_solutions)))  # type: ignore
        else:
            self._active_indices = subset_indices

    def generate(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Generate a decision vector based on user preferences by selecting the top K nearest
        candidate solutions in objective space and computing their weighted average.

        Args:
            preferences (ObjectivePreferences): User's objective preferences (e.g., weights for time and energy).

        Returns:
            NDArray[np.float64]: Recommended decision vector in decision space.

        Raises:
            RuntimeError: If the interpolator has not been fitted.
        """
        if self._candidate_solutions is None or self._objective_front is None:
            raise RuntimeError("Interpolator has not been fitted.")

        weights = np.array([preferences.time_weight, preferences.energy_weight])
        active_objectives = self._objective_front[self._active_indices]  # type: ignore
        similarity_scores = self.similarity_metric(active_objectives, weights)

        # Indices of top-k points with highest similarity scores (descending order)
        top_k_idx = np.argsort(similarity_scores)[-self.k :]
        selected_indices = [self._active_indices[i] for i in top_k_idx]  # type: ignore

        top_k_solutions = self._candidate_solutions[selected_indices]
        top_k_similarities = similarity_scores[top_k_idx]
        sum_sim = np.sum(top_k_similarities)

        if sum_sim < 1e-12:
            # Fallback: mean of the top K if similarity scores are near zero
            return np.mean(top_k_solutions, axis=0)

        weights_normalized = top_k_similarities / sum_sim
        recommendation = np.average(top_k_solutions, axis=0, weights=weights_normalized)
        return recommendation
