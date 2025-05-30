import numpy as np
from numpy.typing import NDArray

from ..preference import ObjectivePreferences
from ..similarities import SimilarityMetric
from .base import BaseInterpolator


class LinearInterpolator(BaseInterpolator):
    """
    Interpolator that recommends a candidate solution by performing linear interpolation
    over a set of Pareto-optimal solutions, weighted by similarity to user preferences.
    """

    def __init__(self, similarity_metric: SimilarityMetric):
        self.similarity_metric = similarity_metric

        # Initialized after fitting
        self.front_positions: NDArray[np.float64] = np.array(
            []
        )  # 1D parameter values (e.g., linspace) to interpolate along the Pareto front
        self.candidate_solutions: NDArray[np.float64] = np.array(
            []
        )  # Decision vectors (inputs)
        self.corresponding_objectives: NDArray[np.float64] = np.array(
            []
        )  # Objective vectors (outputs)

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_vectors: NDArray[np.float64],
    ) -> None:
        """
        Store and preprocess the candidate solutions and their corresponding objective vectors.

        Args:
            candidate_solutions: Array of decision vectors.
            objective_vectors: Array of corresponding objective vectors.
        """
        if len(candidate_solutions) != len(objective_vectors):
            raise ValueError(
                "Candidate solutions and objective vectors must have the same number of entries."
            )

        # Sort by the first objective to ensure a consistent traversal order
        sorted_indices = np.argsort(objective_vectors[:, 0])
        self.candidate_solutions = candidate_solutions[sorted_indices]
        self.corresponding_objectives = objective_vectors[sorted_indices]

        # Generate interpolation parameters along the front (from 0.0 to 1.0)
        num_points = len(candidate_solutions)
        if num_points == 1:
            self.front_positions = np.array(
                [0.5]
            )  # Arbitrary mid-value for single point
        else:
            self.front_positions = np.linspace(0.0, 1.0, num_points)

    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Recommend a candidate solution based on weighted preferences over objectives.

        Args:
            preferences: Preference weights (e.g., time and energy).

        Returns:
            A recommended decision vector interpolated from the Pareto front.
        """
        if len(self.front_positions) == 0:
            raise ValueError("Interpolator has not been fitted with any data.")

        # Convert preferences to a vector of weights
        preference_weights = np.array(
            [preferences.time_weight, preferences.energy_weight]
        )

        # Compute similarity scores between each objective vector and the preferences
        similarity_scores = self.similarity_metric(
            self.corresponding_objectives, preference_weights
        )
        total_similarity = np.sum(similarity_scores)

        # If all similarities are zero, default to midpoint interpolation
        if abs(total_similarity) < 1e-12:
            interpolation_position = float(np.mean(self.front_positions))
        else:
            # Use similarity-weighted average to determine the interpolation position
            interpolation_position = float(
                np.dot(self.front_positions, similarity_scores) / total_similarity
            )

        # Return interpolated decision vector based on the computed position
        return self._interpolate_at(interpolation_position)

    def _interpolate_at(self, position: float) -> NDArray[np.float64]:
        """
        Perform linear interpolation between two candidate solutions based on a given position.

        Args:
            position: Float in [0, 1] indicating where to interpolate along the front.

        Returns:
            Interpolated candidate solution (decision vector).
        """
        if len(self.front_positions) == 1:
            return self.candidate_solutions[0]  # Only one candidate — return as is

        # Locate the first front position greater than or equal to the target
        idx = np.searchsorted(self.front_positions, position)

        if idx == 0:
            return self.candidate_solutions[0]  # Position before first entry
        if idx == len(self.front_positions):
            return self.candidate_solutions[-1]  # Position beyond last entry

        # Extract bounding points and corresponding front positions
        lower_pos, upper_pos = self.front_positions[idx - 1], self.front_positions[idx]
        lower_sol, upper_sol = (
            self.candidate_solutions[idx - 1],
            self.candidate_solutions[idx],
        )

        # Handle potential numerical edge case (e.g., duplicate front positions)
        if upper_pos <= lower_pos:
            return (
                lower_sol if position - lower_pos < upper_pos - position else upper_sol
            )

        # Calculate interpolation factor between 0 and 1
        t = (position - lower_pos) / (upper_pos - lower_pos)
        t = np.clip(t, 0.0, 1.0)

        # Return interpolated solution
        return (1 - t) * lower_sol + t * upper_sol
