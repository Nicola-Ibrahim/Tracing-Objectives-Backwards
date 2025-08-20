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
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """
        Fits the mapper with its knowledge base of known points.
        This concrete method performs universal data validation.

        Subclasses must call `super().fit(objectives, decisions)` at the start of their
        own `fit` method before performing their specific fitting logic.

        Args:
            X (NDArray[np.float64]): Known points in the 'independent' space (features).
            y (NDArray[np.float64]): Corresponding points in the 'dependent' space (targets).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the mapper.")

        # Store dimensions, which will be used by the dimensionality property
        self._objective_dim = X.shape[1]
        self._decision_dim = y.shape[1]

    @abstractmethod
    def predict(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Predicts corresponding 'dependent' values for given feature points.

        Args:
            X (NDArray[np.float64]): The feature points for which to predict targets.
        Returns:
            NDArray[np.float64]: Predicted target values.
        """
        raise NotImplementedError("Predict method not implemented")
