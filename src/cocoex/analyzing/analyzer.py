import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, FieldValidationInfo, field_validator


class ObjectivePreferences(BaseModel):
    """
    Validates and stores user preferences for objective trade-offs.
    Ensures weights are non-negative and sum to 1±epsilon.
    """

    time_weight: float
    energy_weight: float

    @field_validator("time_weight", "energy_weight")
    def validate_individual_weights(cls, v):
        """Ensure each weight is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    @field_validator("energy_weight")
    def validate_weight_sum(cls, v: float, info: FieldValidationInfo) -> float:
        """Ensure total weights sum to 1±0.001"""
        if not info.data:  # Handle case where no data exists yet
            return v

        time_weight = info.data.get("time_weight", 0.0)
        if not np.isclose(v + time_weight, 1.0, atol=0.001):
            raise ValueError("Weights must sum to 1.0 ± 0.001")
        return v


class ParetoPreferenceAnalyzer:
    """
    Analyzes user preferences against Pareto-optimal solutions using cosine similarity.
    Maps preference weights to interpolation parameters in decision space.
    """

    def __init__(
        self,
        objective_vectors: NDArray,
        decision_vectors: NDArray,
        normalized_objectives: NDArray,
    ):
        """
        Initialize analyzer with Pareto-optimal solutions.

        1. Store original objective and decision vectors
        2. Calculate objective bounds for normalization
        3. Normalize objectives to [0,1] range
        """
        # Raw Pareto front/set data
        self.objective_vectors = objective_vectors  # Shape: (n_solutions, n_objectives)
        self.decision_vectors = decision_vectors  # Shape: (n_solutions, n_parameters)

        # Preprocessing steps
        self.objective_bounds = self._calculate_objective_bounds()
        self.normalized_objectives = normalized_objectives

    def _calculate_objective_bounds(self) -> list[NDArray, NDArray]:
        """
        Calculate min/max bounds for each objective.
        Essential for normalization and relative comparisons.
        """
        # Compute across solutions (axis=0) for each objective
        return (
            np.min(self.objective_vectors, axis=0),  # Minimum values per objective
            np.max(self.objective_vectors, axis=0),  # Maximum values per objective
        )

    def _normalize_to_unit_vector(self, weights: NDArray) -> NDArray:
        """
        Convert weights to unit vector (L2 normalization).
        Required for proper cosine similarity calculation.
        """
        l2_norm = np.linalg.norm(weights)
        if l2_norm < 1e-10:  # Handle zero vector case
            return np.zeros_like(weights)
        return weights / l2_norm

    def _validate_preference_weights(self, weights: NDArray) -> None:
        """
        Ensure weights are valid:
        - Correct number of dimensions
        - Non-negative values
        """
        if len(weights) != self.objective_vectors.shape[1]:
            raise ValueError(
                f"Expected {self.objective_vectors.shape[1]} weights, got {len(weights)}"
            )
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

    def calculate_cosine_similarity(self, weights: list[NDArray, NDArray]) -> NDArray:
        """
        Compute cosine similarity between user preferences and Pareto solutions.

        Steps:
        1. Validate input weights
        2. Normalize weights to unit vector
        3. Normalize objectives to unit vectors
        4. Compute dot product (cosine similarity)
        """

        # Convert weights to unit vector (L2 normalization)
        unit_weights = self._normalize_to_unit_vector(weights)

        # Normalize objectives to unit vectors
        # Add small epsilon to avoid division by zero
        objective_norms = (
            np.linalg.norm(self.normalized_objectives, axis=1, keepdims=True) + 1e-10
        )
        unit_objectives = self.normalized_objectives / objective_norms

        # Cosine similarity = dot product of unit vectors
        return np.dot(unit_objectives, unit_weights)

    def find_optimal_candidates_idx(
        self, preferences: ObjectivePreferences, num_candidates: int = 2
    ) -> list[NDArray, NDArray]:
        """
        Identify indices of Pareto solutions closest to user preference.

        Steps:
        1. Compute cosine similarity scores
        2. Sort solutions by similarity
        3. Select top candidates
        """
        # Extract weights from preferences
        weights = np.array([preferences.time_weight, preferences.energy_weight])

        # Get similarity scores for all solutions
        similarity_scores = self.calculate_cosine_similarity(weights)

        # Get indices sorted from least to most similar
        sorted_indices = np.argsort(similarity_scores)

        # Select last 'num_candidates' indices (most similar)
        best_indices = sorted_indices[-num_candidates:]

        return best_indices

    def calculate_interpolation_parameter(
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
        similarity_scores = self.calculate_cosine_similarity(weights)

        # Fallback if no similarity (all scores ~0)
        if np.sum(similarity_scores) < 1e-10:
            return 0.5  # Default to midpoint

        # Create position array [0, 1/(n-1), 2/(n-1), ..., 1]
        n_solutions = self.decision_vectors.shape[0]
        solution_positions = np.linspace(0, 1, n_solutions)

        # Compute weighted average position
        # Higher similarity = stronger influence on position
        weighted_positions = solution_positions * similarity_scores
        return np.sum(weighted_positions) / np.sum(similarity_scores)

    def map_objectives_to_pareto_set(
        self, objective_candidates: NDArray, tolerance: float = 1e-6
    ) -> list[NDArray, NDArray]:
        """
        Map objective space candidates to their corresponding Pareto set solutions.

        Args:
            objective_candidates: Array of objective vectors (n_candidates, n_objectives)
            tolerance: Maximum allowed difference per objective dimension

        Returns:
            list of (indices, decision_vectors) for matched solutions

        Raises:
            ValueError: If any candidate can't be matched to the Pareto front
        """
        indices = []
        for i, candidate in enumerate(objective_candidates):
            # Find all solutions within tolerance
            matches = np.all(
                np.abs(self.objective_vectors - candidate) <= tolerance, axis=1
            )
            match_indices = np.where(matches)[0]

            if not match_indices.size:
                raise ValueError(f"No match found for candidate {i}: {candidate}")

            # Take the first match if multiple
            indices.append(match_indices[0])

        # Return sorted indices and corresponding decision vectors
        sorted_indices = np.sort(indices)
        return sorted_indices, self.decision_vectors[sorted_indices]

    def get_alphas_from_indices(self, indices: NDArray) -> NDArray:
        """
        Convert solution indices to normalized alpha positions [0,1].

        Args:
            indices: Array of solution indices in the Pareto set

        Returns:
            Array of alpha values corresponding to the indices
        """
        n_solutions = self.decision_vectors.shape[0]
        return indices / (n_solutions - 1)
