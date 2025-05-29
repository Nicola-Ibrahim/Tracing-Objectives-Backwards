from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..preference import ObjectivePreferences


class BaseInterpolator(ABC):
    """
    Abstract interface for interpolators that recommend decision vectors based on preferences.
    """

    @abstractmethod
    def fit(
        self,
        decision_vectors: NDArray[np.float64],
        objective_vectors: NDArray[np.float64],
    ) -> None:
        """
        Fit the interpolator with decision and objective vectors.
        """
        pass

    @abstractmethod
    def recommend(self, preferences: ObjectivePreferences) -> NDArray[np.float64]:
        """
        Recommend a decision vector based on user preferences.
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
