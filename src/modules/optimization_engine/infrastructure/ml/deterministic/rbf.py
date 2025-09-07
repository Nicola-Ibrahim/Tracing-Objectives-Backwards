import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator

from ....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
)


class RBFEstimator(DeterministicEstimator):
    def __init__(
        self, kernel: str = "thin_plate_spline", n_neighbors: int = 10
    ) -> None:
        """Initialize the RBF Inverse Decision Mapper."""
        super().__init__()
        self._model: RBFInterpolator = None
        self.neighbors = n_neighbors
        self.kernel = kernel

        valid_kernels = {
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        }
        if kernel not in valid_kernels:
            raise ValueError(
                f"Invalid kernel '{kernel}'. Must be one of {valid_kernels}."
            )
        if n_neighbors < 1:
            raise ValueError("Neighbors must be at least 1.")

    @property
    def type(self) -> str:
        return "RBF"

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs) -> None:
        super().fit(X, y)

        if len(X) < 1:
            raise ValueError(
                "RBF Inverse Decision Mapper requires at least 1 data point for fitting."
            )

        # Combine X and y to find unique rows
        combined_data = np.hstack((X, y))
        unique_data, unique_indices = np.unique(
            combined_data, axis=0, return_index=True
        )

        # Filter X and y to keep only unique data points
        X_unique = X[unique_indices]
        y_unique = y[unique_indices]

        # Check if a singular matrix might still be an issue with a small number of points
        if len(X_unique) < 2 and self.kernel in [
            "thin_plate_spline",
            "cubic",
            "quintic",
        ]:
            raise ValueError(
                f"Kernel '{self.kernel}' requires at least 2 unique data points."
            )

        self._model = RBFInterpolator(
            y=X_unique, d=y_unique, neighbors=self.neighbors, kernel=self.kernel
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._X_dim:
            raise ValueError(
                f"Target objectives must have {self._X_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        return self._model(X)
