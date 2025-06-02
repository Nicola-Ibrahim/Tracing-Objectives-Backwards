from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ...utils.preprocessing.similarities import SimilarityMetric
from ..preference import ObjectivePreferences
from .base import BaseInterpolator


class LinearInterpolator(BaseInterpolator):
    """
    Linear interpolator along a parameterized front. It operates on a subset
    of the fitted data specified during the recommendation call.
    """

    def __init__(self, similarity_metric: SimilarityMetric) -> None:
        """
        Initialize the LinearInterpolator.

        Args:
            similarity_metric (SimilarityMetric): Metric used to compare objective vectors.
        """
        self.similarity_metric = similarity_metric
        # Stores the full dataset the interpolator was fitted on
        self._all_decision_vectors: NDArray[np.float64] | None = None
        self._all_objective_vectors: NDArray[np.float64] | None = None
        self._all_param_coords: NDArray[np.float64] | None = None
        # _initial_sort_indices maps from original input order to internal sorted order
        self._initial_sort_indices: NDArray[np.int_] | None = None

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fits the interpolator with the entire set of candidate solutions and objective vectors.
        This data forms the complete knowledge base for subsequent recommendations.

        Args:
            candidate_solutions (NDArray[np.float64]): Array of all candidate solutions in decision space.
            objective_front (NDArray[np.float64]): Array of all corresponding objective vectors.

        Raises:
            ValueError: If lengths of inputs do not match or are empty.
        """
        if candidate_solutions.shape[0] != objective_front.shape[0]:
            raise ValueError(
                "Candidate solutions and objective front must be of the same length."
            )
        if candidate_solutions.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the interpolator.")

        # Store the original sort order to maintain mapping from input indices
        self._initial_sort_indices = np.argsort(objective_front[:, 0])
        self._all_decision_vectors = candidate_solutions[
            self._initial_sort_indices
        ].astype(np.float64)
        self._all_objective_vectors = objective_front[
            self._initial_sort_indices
        ].astype(np.float64)

        # Create a parameterization from 0 to 1 along the entire sorted front
        n = len(self._all_decision_vectors)
        if n == 1:
            self._all_param_coords = np.array([0.5], dtype=np.float64)
        else:
            self._all_param_coords = np.linspace(0.0, 1.0, n, dtype=np.float64)

    def generate(
        self,
        user_preferences: ObjectivePreferences,
        active_region_indices: Sequence[int] | None = None,
    ) -> NDArray[np.float64]:
        """
        Generates a recommended decision vector by finding a target parameter
        along a specified region of the fitted front based on user preferences,
        and then linearly interpolating the decision vectors.

        Args:
            user_preferences (ObjectivePreferences): User's preferences defining the target in objective space.
            active_region_indices (Sequence[int] | None): Optional. Indices of solutions (from the *original input
                                                      order to fit()*) that define the active region of interest.
                                                      If None, the interpolator uses its entire fitted dataset.

        Returns:
            NDArray[np.float64]: The interpolated decision vector.

        Raises:
            RuntimeError: If the interpolator has not been fitted.
            ValueError: If the specified active region is empty.
        """
        if (
            self._all_decision_vectors is None
            or self._all_objective_vectors is None
            or self._all_param_coords is None
            or self._initial_sort_indices is None
        ):
            raise RuntimeError("Interpolator has not been fitted.")

        # Determine the subset of data to operate on
        if active_region_indices is None:
            # Use the entire fitted dataset if no specific region is provided
            current_decision_vectors = self._all_decision_vectors
            current_objective_vectors = self._all_objective_vectors
            current_param_coords = self._all_param_coords
        else:
            if not active_region_indices:
                raise ValueError("The provided active_region_indices list is empty.")

            # Filter data using active_region_indices (which refer to the original unsorted input to fit())
            # We need to map these original indices to the interpolator's internal sorted order
            # or, more robustly, extract the data corresponding to these indices and then re-sort it.

            # It's safer to re-sort the subset after extraction to guarantee a linear path
            # for the parameterization from 0 to 1 within this specific region.
            # Assuming active_region_indices are valid indices into the *original* data passed to fit.
            # We need to apply _initial_sort_indices to map from original to internal sorted indices.
            # Let's adjust this for clarity: active_region_indices should ideally refer to the
            # indices of the *already sorted* _all_decision_vectors for direct indexing,
            # or we need a more complex mapping.

            # For robust linear interpolation, the selected subset *must* be sorted along one objective.
            # So, we extract the points and then re-sort them for the current operation.

            # Using the original `_all_decision_vectors` and `_all_objective_vectors` which are already sorted
            # based on the first objective from the `fit` method.
            # If `active_region_indices` comes from `ParetoRecommender`'s `_identify_preferred_region_indices`
            # these indices refer to the `self.pareto_objective_front_data` (which is unsorted).
            # So, we extract, then re-sort.

            temp_decision_vectors = self._all_decision_vectors[active_region_indices]
            temp_objective_vectors = self._all_objective_vectors[active_region_indices]

            # Re-sort this subset by its first objective to define a new linear path
            subset_sorted_indices = np.argsort(temp_objective_vectors[:, 0])
            current_decision_vectors = temp_decision_vectors[subset_sorted_indices]
            current_objective_vectors = temp_objective_vectors[subset_sorted_indices]

            # Re-parameterize the selected and sorted subset from 0 to 1
            n_subset = len(current_decision_vectors)
            if n_subset == 1:
                current_param_coords = np.array([0.5], dtype=np.float64)
            else:
                current_param_coords = np.linspace(0.0, 1.0, n_subset, dtype=np.float64)

        if len(current_decision_vectors) == 0:  # Final check after subsetting/sorting
            raise ValueError(
                "No data points available in the active region for recommendation."
            )

        # Convert preferences to a numpy array for similarity calculation
        # Assuming preferences.weights provides the numerical values for alignment
        preference_vector_for_similarity = np.array(
            user_preferences.weights, dtype=np.float64
        )

        # Calculate similarity scores between the objectives in the active region and user preferences
        alignment_scores = self.similarity_metric(
            current_objective_vectors, preference_vector_for_similarity
        )

        total_alignment = np.sum(alignment_scores)

        # Determine the query parameter based on weighted average of parameter coordinates
        if np.isclose(total_alignment, 0.0):
            # If all similarities are near zero, fallback to the mean param coordinate of the active region
            query_param = float(np.mean(current_param_coords))
        else:
            # Weighted average of parameter coordinates, weighted by alignment scores
            query_param = float(
                np.dot(current_param_coords, alignment_scores) / total_alignment
            )

        return self._linear_interpolate(
            query_param, current_decision_vectors, current_param_coords
        )

    def _linear_interpolate(
        self,
        query_param: float,
        decision_vectors_to_use: NDArray[np.float64],
        param_coords_to_use: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Performs linear interpolation of decision vectors based on a given query parameter
        within the provided dataset.

        Args:
            query_param (float): A scalar parameter used to locate a point along the parameterized front.
            decision_vectors_to_use (NDArray[np.float64]): Decision vectors for the active region.
            param_coords_to_use (NDArray[np.float64]): Parameter coordinates for the active region.

        Returns:
            NDArray[np.float64]: Interpolated decision vector.
        """
        n = len(param_coords_to_use)
        if n == 1:
            return decision_vectors_to_use[0]

        # Find the index of the first parameter coordinate greater than or equal to query_param
        idx = np.searchsorted(param_coords_to_use, query_param)

        # Handle edge cases: query_param outside the range [0, 1] or beyond the data points
        if idx == 0:
            return decision_vectors_to_use[
                0
            ]  # Return the first solution if query_param is before the first point
        if idx >= n:
            return decision_vectors_to_use[
                -1
            ]  # Return the last solution if query_param is after the last point

        # Get the two bounding points for interpolation
        left_param, right_param = param_coords_to_use[idx - 1], param_coords_to_use[idx]
        left_solution, right_solution = (
            decision_vectors_to_use[idx - 1],
            decision_vectors_to_use[idx],
        )

        # Defensive check for identical parameter coordinates (shouldn't happen with linspace usually)
        if np.isclose(right_param, left_param):
            # If parameters are identical, return the solution of the closer point
            dist_left = abs(query_param - left_param)
            dist_right = abs(right_param - query_param)
            return left_solution if dist_left < dist_right else right_solution

        # Compute interpolation weight (t)
        # t ranges from 0 to 1 within the segment [left_param, right_param]
        t = (query_param - left_param) / (right_param - left_param)
        t = np.clip(
            t, 0.0, 1.0
        )  # Ensure t is within [0, 1] due to floating point inaccuracies

        # Perform linear interpolation in the decision space
        return (1 - t) * left_solution + t * right_solution
