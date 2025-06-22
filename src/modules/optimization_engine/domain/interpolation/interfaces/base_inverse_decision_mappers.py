from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseInverseDecisionMapper(ABC):
    """
    Base class for Inverse Decision Mappers (and general interpolators in this context).
    It defines the common interface for fitting the mapper and making predictions.

    In the context of ParetoDataService's general interpolation needs:
    - 'objective_space_points' acts as the independent variable (x-axis data).
    - 'decision_space_points' acts as the dependent variable (y-axis data).
    - 'target_objective_points' are the points at which to evaluate the interpolation.

    For specific inverse mapping use cases (e.g., objectives -> decisions):
    - 'objective_space_points' would indeed be objective values.
    - 'decision_space_points' would be corresponding decision values.
    - 'target_objective_points' would be new objective values for which to find decisions.
    """

    @abstractmethod
    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        """
        Fits the mapper with its knowledge base of known points.

        Args:
            objective_space_points (NDArray[np.float64]): Known points in the 'independent' space.
            decision_space_points (NDArray[np.float64]): Corresponding points in the 'dependent' space.
        """
        if len(objective_space_points) != len(decision_space_points):
            raise ValueError(
                "objective_space_points and decision_space_points must have the same number of samples."
            )
        if objective_space_points.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the mapper.")

    @abstractmethod
    def predict(
        self,
        target_objective_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Predicts corresponding 'dependent' values for given 'independent' target points.

        Args:
            target_objective_points (NDArray[np.float64]): The points in the 'independent' space
                                                          for which to predict 'dependent' values.
        Returns:
            NDArray[np.float64]: Predicted values in the 'dependent' space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensionality(self) -> str:
        """Returns the dimensionality handled by the mapper (e.g., '1D', 'ND')."""
        pass


class Base1DInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    Base class for 1-Dimensional Inverse Decision Mappers.
    Expects 1D arrays for objective_space_points and decision_space_points during fit.
    """

    @property
    def dimensionality(self) -> str:
        return "1D"

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        if objective_space_points.ndim > 1 or decision_space_points.ndim > 1:
            raise ValueError(
                "1D mappers expect 1D arrays for objective_space_points and decision_space_points."
            )
        super().fit(objective_space_points, decision_space_points)


class BaseNDInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    Base class for N-Dimensional Inverse Decision Mappers.
    Expects objective_space_points to be 2D (n_samples, n_features_x) and
    decision_space_points to be 1D (n_samples,) or 2D (n_samples, n_features_y).
    """

    @property
    def dimensionality(self) -> str:
        return "ND"

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        if objective_space_points.ndim < 1:
            raise ValueError(
                "ND mappers expect objective_space_points to be at least 1D (or 2D for multiple features)."
            )
        super().fit(objective_space_points, decision_space_points)
