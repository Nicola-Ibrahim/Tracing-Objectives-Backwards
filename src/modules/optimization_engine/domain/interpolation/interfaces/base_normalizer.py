from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


class BaseNormalizer(BaseEstimator, TransformerMixin):
    """Base class for all normalizers following sklearn interface"""

    def fit(self, X: NDArray[np.float64], y: NDArray | None = None) -> Self:
        """
        Compute and store normalization Args from the training data.

        Args:
            X: Input data array (n_samples, n_features)
            y: Optional target values (ignored)

        Returns:
            The fitted normalizer instance (self)
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply the learned normalization to new data.

        Args:
            X: Data to be transformed

        Returns:
            Normalized version of the input data
        """
        raise NotImplementedError("Subclasses must implement transform method")

    def fit_transform(
        self, X: NDArray[np.float64], y: NDArray | None = None
    ) -> NDArray[np.float64]:
        """
        Fit to data, then transform it.
        Equivalent to fit().transform() but more efficient.

        Args:
            X: Input data array (n_samples, n_features)
            y: Optional target values (ignored)

        Returns:
            Normalized version of the input data
        """
        raise NotImplementedError("Subclasses must implement fit_transform method")

    def inverse_transform(self, X_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Reverse the normalization to reconstruct original data.

        Args:
            X_norm: Normalized data

        Returns:
            Data in the original scale
        """
        raise NotImplementedError("Subclasses must implement inverse_transform method")

    def fit_transform(
        self, X: NDArray[np.float64], y: NDArray | None = None
    ) -> NDArray[np.float64]:
        """
        Convenience method: fit to data, then transform it.
        Equivalent to fit().transform() but more efficient.
        """
        self.fit(X, y)
        return self.transform(X)
