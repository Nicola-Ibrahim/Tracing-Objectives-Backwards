from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseInterpolator(ABC):
    """
    Base class for all interpolators.
    """

    def __init__(self, positions: NDArray[np.float_], solutions: NDArray[np.float_]):
        """
        Initialize the base interpolator.
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the interpolator to the provided data.

        Args:
            *args: Positional arguments for fitting.
            **kwargs: Keyword arguments for fitting.
        Returns:
            None
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def interpolate(self, x):
        """
        Interpolate the value at x.

        Args:
            x: The input value to interpolate.
        Returns:
            Interpolated value.
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
