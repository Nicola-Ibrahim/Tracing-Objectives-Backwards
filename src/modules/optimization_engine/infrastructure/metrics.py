from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_squared_error as sk_mean_squared_error,
)

from ..domain.interpolation.interfaces.base_metric import BaseValidationMetric


class MeanSquaredErrorValidationMetric(BaseValidationMetric):
    """
    Concrete implementation of BaseValidationMetric for Mean Squared Error (MSE).
    Leverages scikit-learn's mean_squared_error for robust calculation.
    """

    def calculate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> float:
        """
        Calculates the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true: A NumPy array of true (actual) values.
            y_pred: A NumPy array of predicted values. Must have the same shape as y_true.

        Returns:
            The Mean Squared Error as a float.
        """
        # Ensure inputs are numpy arrays if they aren't already (though type hints help)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic shape check for compatibility, sklearn will also handle this
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        return float(sk_mean_squared_error(y_true, y_pred))


class MetricFactory:
    """
    Concrete factory for creating various validation metric instances.
    """

    def create(self, config: dict) -> BaseValidationMetric:
        """
        Creates and returns a concrete validation metric instance based on the given type and parameters.
        """
        metric_type = config.get("type")
        params = config.get("params", {})

        if metric_type == "MSE":
            return MeanSquaredErrorValidationMetric(**params)

        # elif metric_type == "MAE":
        #     return MeanAbsoluteErrorValidationMetric(**params)

        # elif metric_type == "R2":
        #     return R2ScoreValidationMetric(**params)

        # elif metric_type == "MAPE":
        #     return MeanAbsolutePercentageErrorValidationMetric(**params)

        # elif metric_type == "SMAPE":
        #     return SymmetricMeanAbsolutePercentageErrorValidationMetric(**params)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
