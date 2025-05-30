import numpy as np
from numpy.typing import NDArray

from .interpolators.base import BaseInterpolator
from .preference import ObjectivePreferences


class ParetoRecommender:
    """
    Recommender for Pareto-optimal candidate solutions based on user-defined objective preferences.

    This class delegates interpolation logic to a pluggable strategy (interpolator),
    which determines how to traverse the Pareto front and recommend a suitable solution.
    """

    def __init__(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
        interpolator: BaseInterpolator,
    ):
        """
        Initialize the Pareto recommender with a set of candidate solutions and their corresponding objectives.

        Args:
            candidate_solutions (NDArray[np.float64]): Array of decision vectors representing possible solutions.
            objective_front (NDArray[np.float64]): Array of objective vectors representing outcomes for each solution.
            interpolator (BaseInterpolator): Strategy used to interpolate between solutions based on preferences.
        """
        if not isinstance(candidate_solutions, np.ndarray) or not isinstance(
            objective_front, np.ndarray
        ):
            raise TypeError("Inputs must be NumPy arrays.")

        if candidate_solutions.ndim < 2:
            candidate_solutions = candidate_solutions.reshape(-1, 1)
        if objective_front.ndim < 2:
            objective_front = objective_front.reshape(-1, 1)

        if len(candidate_solutions) != len(objective_front):
            raise ValueError(
                "Each candidate solution must have a corresponding objective vector."
            )

        self.interpolator = interpolator
        self.interpolator.fit(candidate_solutions, objective_front)

    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Recommend a candidate solution based on user preferences over objectives.

        Args:
            preferences (ObjectivePreferences): Weights or priorities assigned to each objective.

        Returns:
            NDArray[np.float64]: A recommended decision vector from the candidate solution set.
        """
        return self.interpolator.recommend(preferences)
