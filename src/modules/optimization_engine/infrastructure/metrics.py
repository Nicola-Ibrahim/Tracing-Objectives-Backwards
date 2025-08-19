import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score

from ..domain.model_evaluation.interfaces.base_metric import BaseValidationMetric


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


class MeanAbsoluteErrorValidationMetric(BaseValidationMetric):
    """
    Concrete implementation of BaseValidationMetric for Mean Absolute Error (MAE).
    Leverages scikit-learn's mean_absolute_error for robust calculation.
    """

    def calculate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> float:
        """
        Calculates the Mean Absolute Error (MAE) between true and predicted values.

        Args:
            y_true: A NumPy array of true (actual) values.
            y_pred: A NumPy array of predicted values. Must have the same shape as y_true.

        Returns:
            The Mean Absolute Error as a float.
        """

        # Ensure inputs are numpy arrays if they aren't already
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic shape check for compatibility, sklearn will also handle this
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        return float(sk_mean_absolute_error(y_true, y_pred))


class R2ScoreValidationMetric(BaseValidationMetric):
    """
    Concrete implementation of BaseValidationMetric for R^2 Score.
    Leverages scikit-learn's r2_score for robust calculation.
    """

    def calculate(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> float:
        """
        Calculates the R^2 Score between true and predicted values.

        Args:
            y_true: A NumPy array of true (actual) values.
            y_pred: A NumPy array of predicted values. Must have the same shape as y_true.

        Returns:
            The R^2 Score as a float.
        """
        # Ensure inputs are numpy arrays if they aren't already
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic shape check for compatibility, sklearn will also handle this
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        return float(sk_r2_score(y_true, y_pred))
