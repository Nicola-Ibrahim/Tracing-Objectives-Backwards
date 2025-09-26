import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

from .....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
)


class LinearNDEstimator(DeterministicEstimator):
    def __init__(self) -> None:
        super().__init__()
        self._interp_func: LinearNDInterpolator | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        # 1. Call the parent's fit method for shared validation and dimension storage
        super().fit(X, y)

        # 2. Perform specific validation for this interpolator
        if len(X) < 3:
            raise ValueError(
                "LinearNDEstimator requires at least 3 data points for fitting."
            )

        # 3. Perform specific fitting logic
        self._interp_func = LinearNDInterpolator(points=X, values=y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._X_dim:
            raise ValueError(
                f"Target objectives must have {self._X_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        # Call the fitted interpolation function
        return self._interp_func(X)
