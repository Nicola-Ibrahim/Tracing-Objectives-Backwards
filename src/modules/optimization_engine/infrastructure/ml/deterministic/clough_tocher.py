import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator

from ....domain.model_management.interfaces.base_ml_mapper import (
    DeterministicMlMapper,
)


class CloughTocherMlMapper(DeterministicMlMapper):
    def __init__(self) -> None:
        super().__init__()
        self._interp_func: CloughTocher2DInterpolator | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the interpolator using features X and targets y."""
        super().fit(X, y)

        if X.shape[1] != 2:
            raise ValueError("CloughTocherMlMapper requires 2D objective data.")
        if len(X) < 4:
            raise ValueError(
                "CloughTocherMlMapper requires at least 4 data points for fitting."
            )

        self._interp_func = CloughTocher2DInterpolator(
            points=X, values=y, fill_value="extrapolate"
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, but got {X.shape[1]} dimensions."
            )

        return self._interp_func(X)
