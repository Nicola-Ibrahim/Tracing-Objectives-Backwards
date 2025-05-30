from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..preference import ObjectivePreferences


class BaseInterpolator(ABC):
    """
    Abstract base class for interpolators that recommend candidate solutions
    based on user preferences over objective values.
    """

    @abstractmethod
    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fit the interpolator with candidate solutions and their corresponding objective vectors.

        Args:
            candidate_solutions (NDArray): Array of decision vectors.
            objective_front (NDArray): Array of corresponding objective vectors.
        """
        pass

    @abstractmethod
    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Recommend a candidate solution based on user-defined preferences.

        Args:
            preferences (ObjectivePreferences): Preference weights over objectives.

        Returns:
            NDArray: A single recommended decision vector.
        """
        pass


class PreferenceDrivenInterpolator(BaseInterpolator):
    """
    Base class for preference-driven interpolators that use user-defined
    objective weights to determine the best interpolation parameter.
    This class extends the BaseInterpolator with methods to compute
    the preferred parameter coordinate based on similarity scores.
    """

    @abstractmethod
    def select_parameter_position(
        self,
        parameter_positions: NDArray[np.float64],
        similarity_scores: NDArray[np.float64],
    ) -> float:
        pass
