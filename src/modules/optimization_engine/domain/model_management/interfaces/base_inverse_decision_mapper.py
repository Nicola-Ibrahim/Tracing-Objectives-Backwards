from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseInverseDecisionMapper(ABC):
    """
    Base class for Inverse Decision Mappers (and general interpolators in this context).
    It defines the common interface for fitting the mapper and making predictions.

    This class now enforces a clean abstraction by providing shared logic in concrete methods
    and requiring subclasses to implement core functionality.
    """

    def __init__(self) -> None:
        """
        Initializes the base inverse decision mapper.
        Subclasses should call this constructor via super().__init__().
        """
        self._objective_dim: int | None = None
        self._decision_dim: int | None = None

    @property
    def dimensionality(self) -> str:
        """
        Returns the dimensionality handled by the mapper.
        This is determined from the input data during fitting.
        """
        if self._objective_dim is None:
            return "Unfitted"  # A better state than 'ND'
        elif self._objective_dim == 1:
            return "1D"
        else:
            return "ND"

    @property
    def type(self) -> str:
        """
        Returns the type of the inverse decision mapper.
        This should be overridden by subclasses to return their specific type.
        """
        return self.__class__.__name__

    def fit(
        self,
        objectives: npt.NDArray[np.float64],
        decisions: npt.NDArray[np.float64],
    ) -> None:
        """
        Fits the mapper with its knowledge base of known points.
        This concrete method performs universal data validation.

        Subclasses must call `super().fit(objectives, decisions)` at the start of their
        own `fit` method before performing their specific fitting logic.

        Args:
            objectives (NDArray[np.float64]): Known points in the 'independent' space.
            decisions (NDArray[np.float64]): Corresponding points in the 'dependent' space.
        """
        if objectives.ndim == 1:
            objectives = objectives.reshape(-1, 1)
        if decisions.ndim == 1:
            decisions = decisions.reshape(-1, 1)

        if objectives.shape[0] != decisions.shape[0]:
            raise ValueError(
                "objectives and decisions must have the same number of samples."
            )
        if objectives.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the mapper.")

        # Store dimensions, which will be used by the dimensionality property
        self._objective_dim = objectives.shape[1]
        self._decision_dim = decisions.shape[1]

    @abstractmethod
    def predict(
        self,
        target_objectives: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Predicts corresponding 'dependent' values for given 'independent' target points.

        Args:
            target_objectives (NDArray[np.float64]): The points in the 'independent' space
                                                          for which to predict 'dependent' values.
        Returns:
            NDArray[np.float64]: Predicted values in the 'dependent' space.
        """
        raise NotImplementedError  # No implementation in the abstract class
