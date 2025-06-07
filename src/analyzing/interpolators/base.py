from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..domain.preference import ObjectivePreferences


class BaseInterpolator(ABC):
    """
    Base class for interpolators.
    An interpolator is responsible for generating a new decision vector
    from a set of known solutions and their corresponding objective values.
    """

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

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fits the interpolator with its entire knowledge base of solutions.
        This method should be called once before any recommendation is made.
        Args:
            candidate_solutions (NDArray[np.float64]): Known solutions in the decision space.
            objective_front (NDArray[np.float64]): Corresponding objective values in the objective space.
        Returns:
            None
        """

        if len(candidate_solutions) != len(objective_front):
            raise ValueError(
                "candidate_solutions and objective_front must have the same length"
            )

        if candidate_solutions.shape[0] != objective_front.shape[0]:
            raise ValueError(
                "Candidate solutions and objective front must be of the same length."
            )
        if candidate_solutions.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the interpolator.")

    @abstractmethod
    def generate(
        self,
        preferences: ObjectivePreferences,
        active_region_indices: Sequence[int] | None = None,
    ) -> NDArray[np.float64]:
        """
        Generates a new interpolated point based on user preferences and optionally a set of active region indices.
        If active_region_indices is provided, the recommendation is generated
        only from the solutions corresponding to those indices.

        Args:
            preferences (ObjectivePreferences): User preferences for the objectives.
            active_region_indices (Sequence[int] | None): Optional indices of active regions to consider.
        Returns:
            NDArray[np.float64]: A new decision vector generated based on the preferences.
        """
        raise NotImplementedError
