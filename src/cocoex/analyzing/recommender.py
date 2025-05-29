import numpy as np
from numpy.typing import NDArray

from .interpolators.base import BaseInterpolator
from .preference import ObjectivePreferences


class ParetoFrontRecommender:
    """
    Recommender for Pareto-optimal solutions based on decision and objective vectors.
    This class uses an interpolator to suggest solutions based on user preferences
    and the Pareto front defined by decision and objective vectors.
    """

    def __init__(
        self,
        decision_vectors: NDArray[np.float64],
        objective_vectors: NDArray[np.float64],
        interpolator: BaseInterpolator,
    ):
        if not isinstance(decision_vectors, np.ndarray) or not isinstance(
            objective_vectors, np.ndarray
        ):
            raise TypeError("Inputs must be NumPy arrays.")

        if decision_vectors.ndim < 2:
            decision_vectors = decision_vectors.reshape(-1, 1)
        if objective_vectors.ndim < 2:
            objective_vectors = objective_vectors.reshape(-1, 1)

        if len(decision_vectors) != len(objective_vectors):
            raise ValueError(
                "Decision and objective vectors must have the same number of entries."
            )

        self.interpolator = interpolator
        self.interpolator.fit(decision_vectors, objective_vectors)

    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Recommend a decision vector based on preferences.
        Delegates to the interpolator strategy.
        """
        return self.interpolator.recommend(preferences)
