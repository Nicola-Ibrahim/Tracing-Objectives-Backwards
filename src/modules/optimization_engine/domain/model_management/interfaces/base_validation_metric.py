from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class MetricResult(BaseModel):
    name: str
    value: float
    details: dict[str, Any] = None


class BaseValidationMetric(ABC):
    """
    Abstract Base Class for validation metric calculation strategies.
    Defines the interface for any metric calculation method.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the validation metric.

        Returns:
            The name of the metric as a string.
        """
        return self.__class__.__name__

    @abstractmethod
    def calculate(
        self,
        y_true: np.typing.NDArray[np.floating],
        y_pred: np.typing.NDArray[np.floating],
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
