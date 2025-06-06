from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..preference import ObjectivePreferences


class BaseInterpolator(ABC):
    """
    Base class for interpolators.
    An interpolator is responsible for generating a new decision vector
    from a set of known solutions and their corresponding objective values.
    """

    @abstractmethod
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
        raise NotImplementedError

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
