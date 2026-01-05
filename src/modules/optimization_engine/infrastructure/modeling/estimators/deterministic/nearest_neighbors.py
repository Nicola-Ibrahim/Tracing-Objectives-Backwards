import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import NearestNDInterpolator

from .....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
)
from .....domain.modeling.value_objects.estimator_params import (
    NearestNeighborsEstimatorParams,
)


class NearestNDEstimator(DeterministicEstimator):
    _interp_func: NearestNDInterpolator | None = None

    def __init__(self, params: NearestNeighborsEstimatorParams) -> None:
        super().__init__()
        self.params = params

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        # Ensure X is 2D (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        super().fit(X, y)  # Call parent validation
        if len(X) < 1:
            raise ValueError(
                "NearestNDEstimator requires at least 1 data point for fitting."
            )
        self._interp_func = NearestNDInterpolator(x=X, y=y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # Ensure X is 2D (n_samples, n_features) for ND interpolators
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._interp_func(X)
