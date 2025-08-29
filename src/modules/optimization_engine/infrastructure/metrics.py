import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score

from ..domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)


class MeanSquaredErrorValidationMetric(BaseValidationMetric):
    """
    Concrete implementation of BaseValidationMetric for Mean Squared Error (MSE).
    Leverages scikit-learn's mean_squared_error for robust calculation.
    """

    @property
    def name(self) -> str:
        return "MSE"

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

    @property
    def name(self) -> str:
        return "MAE"

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

    @property
    def name(self) -> str:
        return "R2"

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


class NegativeLogLikelihoodMetric(BaseValidationMetric):
    """
    Calculates the Negative Log-Likelihood (NLL) for a probabilistic model.
    This is the appropriate metric for evaluating Mixture Density Networks.
    """

    @property
    def name(self) -> str:
        return "Negative Log-Likelihood"

    def calculate(
        self,
        y_true: NDArray[np.float64],
        y_pred: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> float:
        """
        Calculates the NLL.

        Args:
            y_true: A NumPy array of true values, shape (n_samples, n_outputs).
            y_pred: A tuple containing the MDN's output parameters:
                - mixing_coeffs: (n_samples, n_mixtures)
                - means: (n_samples, n_mixtures, n_outputs)
                - variances: (n_samples, n_mixtures, n_outputs)

        Returns:
            The average Negative Log-Likelihood as a float.
        """
        mixing_coeffs, means, variances = y_pred

        # Handle the 1D objective case where `y_true` might be (n_samples, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        n_samples = y_true.shape[0]
        n_mixtures = mixing_coeffs.shape[1]

        # Initialize an array to store the log-likelihood for each sample
        sample_log_probs = np.zeros(n_samples)

        for i in range(n_samples):
            # The likelihood for a single sample is a weighted sum of the
            # probability densities of each Gaussian component.
            mixture_likelihood = 0.0
            for j in range(n_mixtures):
                # Get the parameters for this specific Gaussian component
                mixture_weight = mixing_coeffs[i, j]
                mixture_mean = means[i, j, :]
                mixture_variance = variances[i, j, :]

                # Use a multivariate Gaussian PDF
                # We need to reshape the variance for the `multivariate_normal` function
                # as it expects a 1D array for diagonal covariance.
                pdf = multivariate_normal.pdf(
                    y_true[i, :], mean=mixture_mean, cov=np.diag(mixture_variance)
                )

                mixture_likelihood += mixture_weight * pdf

            # Add a small epsilon to avoid taking the log of zero
            sample_log_probs[i] = np.log(mixture_likelihood + 1e-10)

        # The NLL is the negative of the sum of the log-likelihoods
        # (or the negative mean, if you want a per-sample average)
        nll = -np.mean(sample_log_probs)

        return nll
