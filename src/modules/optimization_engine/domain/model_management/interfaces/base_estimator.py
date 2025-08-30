import enum
import inspect
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt


class BaseEstimator(ABC):
    """
    Base class for Inverse Decision Mappers (and general interpolators in this context).
    It defines the common interface for fitting the mapper and making predictions.

    This class now enforces a clean abstraction by providing shared logic in concrete methods
    and requiring subclasses to implement core functionality.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the base inverse decision mapper.
        Subclasses should call this constructor via super().__init__().
        """

        # Save constructor args for metadata
        sig = inspect.signature(self.__class__.__init__)
        bound = sig.bind_partial(self, **kwargs)
        bound.apply_defaults()
        self._init_params = {k: v for k, v in bound.arguments.items() if k != "self"}

        self._X_dim: int | None = None
        self._y_dim: int | None = None

        self._training_history: dict | None = None

    @property
    def dimensionality(self) -> str:
        """
        Returns the dimensionality handled by the mapper.
        This is determined from the input data during fitting.
        """
        if self._X_dim is None:
            return "Unfitted"  # A better state than 'ND'
        elif self._X_dim == 1:
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
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], **kwargs
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

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the mapper.")

        # Store dimensions, which will be used by the dimensionality property
        self._X_dim = X.shape[1]
        self._y_dim = y.shape[1]

    def to_dict(self):
        return {k: self._serialize(v) for k, v in self._init_params.items()}

    @classmethod
    def _serialize(cls, v):
        if isinstance(v, enum.Enum):
            return v.value
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, BaseEstimator):
            return v.to_dict()
        elif isinstance(v, list):
            return [cls._serialize(i) for i in v]
        elif isinstance(v, dict):
            return {kk: cls._serialize(vv) for kk, vv in v.items()}
        else:
            return v

    def clone(self) -> Self:
        """
        Clones an estimator by re-instantiating it with the same __init__ parameters.
        """
        klass = self.__class__

        # Get the signature of __init__
        sig = inspect.signature(klass.__init__)
        bound_args = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # Check if the attribute exists with the same name or as a private variable
            if hasattr(self, name):
                bound_args[name] = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                bound_args[name] = getattr(self, f"_{name}")
            else:
                bound_args[name] = param.default

        # Create a new instance with the copied parameters
        return klass(**bound_args)

    def get_loss_history(self) -> dict[str, list]:
        return self._training_history


class DeterministicEstimator(BaseEstimator):
    """
    A deterministic inverse decision mapper that uses a fixed mapping strategy.
    """

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


class ProbabilisticEstimator(BaseEstimator):
    @abstractmethod
    def predict(
        self, X: npt.NDArray[np.float64], n_samples: int
    ) -> npt.NDArray[np.float64]:
        """
        Predicts corresponding 'dependent' values for given feature points.

        Args:
            X (NDArray[np.float64]): The feature points for which to predict targets.
            mode (str): The prediction mode (e.g., "mean", "sample").
        Returns:
            NDArray[np.float64]: Predicted target values.
        """
        raise NotImplementedError("Predict method not implemented")
