from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseValidationMetric(ABC):
    """
    Abstract Base Class for validation metric calculation strategies.
    Defines the interface for any metric calculation method.
    """

    @abstractmethod
    def calculate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> float:
        """
        Calculates a validation metric given true and predicted values.

        Args:
            y_true: A NumPy array of true (actual) values.
            y_pred: A NumPy array of predicted values.

        Returns:
            The calculated metric value as a float.
        """
        pass
