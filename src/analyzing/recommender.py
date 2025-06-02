import numpy as np
from interpolators.base import BaseInterpolator
from numpy.typing import NDArray
from preference import ObjectivePreferences
from deprecated.utils.similarities import SimilarityMetric


class ParetoRecommender:
    """
    Recommender that suggests decision vectors based on user preferences and a set of candidate solutions.
    This class uses an interpolator to derive decision vectors from a set of candidate solutions
    and their corresponding objective values.
    """

    def __init__(
        self,
        candidate_solutions: NDArray[np.float32],
        objective_front: NDArray[np.float32],
        interpolator: BaseInterpolator,
        similarity_metric: SimilarityMetric,
    ) -> None:
        if len(candidate_solutions) != len(objective_front):
            raise ValueError(
                "candidate_solutions and objective_front must have the same length"
            )

        if candidate_solutions.shape[0] != objective_front.shape[0]:
            raise ValueError(
                "Number of decision solutions must match the number of objective value sets."
            )
        if candidate_solutions.shape[0] == 0:
            raise ValueError(
                "Input decision solutions and objective values cannot be empty."
            )

        self.candidate_solutions = candidate_solutions
        self.objective_front = objective_front
        self.interpolator = interpolator
        self.similarity_metric = similarity_metric
        self._is_fitted = False

        self._is_initialized = False
        # The recommender initializes the interpolator with its full knowledge base ONCE
        self._initialize_recommender_components()

    def _initialize_recommender_components(self) -> None:
        """
        Initializes and fits the internal components of the recommender,
        specifically fitting the solution interpolator with the full dataset.
        This operation is performed only once.
        """
        # The interpolator is fitted with the entire set of known solutions
        self.interpolator.fit(self.candidate_solutions, self.objective_front)
        self._is_initialized = True

    def _derive_param_coords(
        self,
        filtered_objectives: NDArray[np.float32],
        preference_vector: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Derives scalar parameter coordinates by projecting objectives onto the preference vector.

        Args:
            filtered_objectives: Objective vectors of filtered points.
            preference_vector: User preference vector.

        Returns:
            NDArray[np.float32]: Parameter coordinates for interpolation.
        """
        return filtered_objectives @ preference_vector

    def _assess_solution_alignment(
        self, preferences: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Calculates how well each solution on the Pareto front aligns with the
        user's normalized preference vector using the defined alignment metric.

        Args:
            preferences: The user's normalized preference vector.

        Returns:
            NDArray[np.float32]: An array of alignment scores for each solution in `pareto_objective_front_data`.
        """
        return self.similarity_metric(self.objective_front, preferences)

    def _identify_active_region_indices(
        self, alignment_scores: NDArray[np.float32], region_size: int
    ) -> NDArray[np.int_]:
        """
        Identifies the indices of the 'region_size' solutions from the full Pareto front
        that are most aligned with the user's preferences. These indices define the
        "Region of Interest" for the interpolator.

        Args:
            alignment_scores: Scores indicating how well each solution aligns with preferences.
            region_size: The number of top-aligned solutions to include in the region of interest.

        Returns:
            NDArray[np.int_]: Global indices of solutions belonging to the preferred region.
                              Sorted from most to least aligned.
        Raises:
            ValueError: If region_size is not positive.
        """
        if region_size <= 0:
            raise ValueError("region_size must be a positive integer.")
        if region_size > len(alignment_scores):
            region_size = len(alignment_scores)
            # print(f"Warning: Requested region_size ({region_size}) exceeds available solutions. Using all {region_size} available solutions.")

        top_indices = np.argsort(alignment_scores)[-region_size:][::-1]
        return top_indices

    def recommend(
        self, preferences: ObjectivePreferences, region_size: int
    ) -> NDArray[np.float32]:
        """
        Generates a recommended decision solution by first identifying a region of interest
        on the Pareto front based on user preferences, and then interpolating within that region.

        Args:
            preferences (ObjectivePreferences): User's preferences across the objectives.
            region_size (int): The number of top-aligned solutions to define the
                                         region of interest. The interpolator will then be
                                         instructed to operate only on this defined region.

        Returns:
            NDArray[np.float32]: The interpolated decision solution (vector) in the decision space.

        Raises:
            RuntimeError: If the recommender has not been initialized.
            ValueError: If the identified region of interest is empty.
        """
        if not self._is_initialized:
            raise RuntimeError("ParetoRecommender has not been initialized.")

        # Stage 1: Identify the "Preferred Region" in the Objective Space
        alignment_scores = self._assess_solution_alignment(preferences)
        active_region_indices = self._identify_active_region_indices(
            alignment_scores, region_size
        )

        if len(active_region_indices) == 0:
            raise ValueError(
                "No solutions found within the preferred region to perform interpolation."
            )

        # Stage 2: Delegate the final recommendation generation to the interpolator.
        # The interpolator's `recommend` method now directly takes the indices
        # of the active region.
        recommended_solution = self.interpolator.generate(
            preferences, active_region_indices=active_region_indices
        )

        return recommended_solution
