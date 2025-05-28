import numpy as np
from numpy.typing import NDArray

from ..utils.data import normalize_to_hypercube
from .interpolators.base import BaseInterpolator
from .preference import ObjectivePreferences
from .similarities import SimilarityMetric


class ParetoAnalyzer:
    def __init__(
        self,
        solutions: NDArray[np.float_],
        front: NDArray[np.float],
        interpolator: BaseInterpolator,
        similarity_metric: SimilarityMetric,
        sort_by_obj: int = 0,
    ) -> None:
        # Sort by one objective to establish order along the front
        order = np.argsort(front[:, sort_by_obj])
        self.solutions = solutions[order]
        self.front = front[order]

        # Store [min,max] for each objective (bounds)
        self.bounds = self._compute_bounds(self.front)

        # Scale front to hypercube [0,1]^d
        self.scaled_front = normalize_to_hypercube(self.front)

        # Prepare interpolation: uniform positions along the sorted front
        n_pts = len(self.solutions)
        self.positions = np.linspace(0.0, 1.0, n_pts)
        self.interpolator = interpolator
        self.interpolator.fit(self.positions, self.solutions)  # type: ignore

        # Similarity function (e.g. cosine)
        self.similarity_metric = similarity_metric

    def _compute_bounds(self) -> list[NDArray, NDArray]:
        """Compute min and max per column for normalization or checks."""
        return (
            np.min(self.front, axis=0),  # Minimum values per objective
            np.max(self.front, axis=0),  # Maximum values per objective
        )

    def top_candidate_indices(
        self, preferences: ObjectivePreferences, k: int = 2
    ) -> NDArray[np.int_]:
        """
        Return indices of the k most similar Pareto points to given weights.

        Args:
            preferences: User-defined weights for objectives (time, energy).
            k: Number of top candidates to return.

        Returns:
            List of indices corresponding to the k most similar solutions.
        """
        # Extract weights from preferences
        weights = np.array([preferences.time_weight, preferences.energy_weight])

        # Get similarity scores for all solutions
        similarity_scores = self.similarity_metric(array=self.front, vector=weights)

        return np.argsort(similarity_scores)[-k:]

    def calculate_interpolator_parameter(
        self, preferences: ObjectivePreferences
    ) -> float:
        """
        Convert preferences to α ∈ [0,1] using weighted average of positions.

        Steps:
        1. Compute similarity scores
        2. Handle edge case with all-zero scores
        3. Map solutions to normalized positions
        4. Compute similarity-weighted average position
        """

        # Extract weights from preferences
        weights = np.array([preferences.time_weight, preferences.energy_weight])

        # Calculate cosine similarity scores
        similarity_scores = self.similarity_metric(array=self.front, vector=weights)

        # Fallback if no similarity (all scores ~0)
        if np.sum(similarity_scores) < 1e-10:
            return 0.5  # Default to midpoint

        # Create position array [0, 1/(n-1), 2/(n-1), ..., 1]
        n_solutions = self.solutions.shape[0]
        solution_positions = np.linspace(0, 1, n_solutions)

        # Compute weighted average position
        # Higher similarity = stronger influence on position
        weighted_positions = solution_positions * similarity_scores
        return np.sum(weighted_positions) / np.sum(similarity_scores)
