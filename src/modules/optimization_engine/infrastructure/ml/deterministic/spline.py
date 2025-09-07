import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import SmoothBivariateSpline

from ....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
)


class SplineEstimator(DeterministicEstimator):
    """
    Inverse Decision Mapper using SciPy's SmoothBivariateSpline for 2D objective spaces.
    """

    _spline_funcs: list[SmoothBivariateSpline] | None = None

    def __init__(self, s: float = 0.0) -> None:
        """
        Initializes the Spline mapper.
        Args:
            s (float): Positive smoothing factor. 0 for interpolation.
        """
        super().__init__()
        self.s = s

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        # 1. Call the parent's fit method for universal validation
        super().fit(X, y)

        # 2. Perform specific validation for this interpolator
        if self._X_dim != 2:
            raise ValueError(
                "SplineEstimator requires features X with exactly 2 dimensions (x, y)."
            )

        # 3. Fit a separate spline for each output dimension.
        # This is necessary because SmoothBivariateSpline can only handle a single output (z).
        self._spline_funcs = []
        for i in range(self._y_dim):
            # SmoothBivariateSpline expects 1D arrays for x, y, z
            spline_func = SmoothBivariateSpline(
                x=X[:, 0],
                y=X[:, 1],
                z=y[:, i],  # Pass only one decision dimension as the output
                s=self.s,
            )
            self._spline_funcs.append(spline_func)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._spline_funcs is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        # If a single point is passed as 1D array, reshape to (1, d)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self._X_dim:
            raise ValueError(
                f"Input X must have {self._X_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        # Call each fitted spline and stack the results
        predictions = []
        for spline_func in self._spline_funcs:
            # `ev` (evaluate) method for bivariate splines
            pred_values = spline_func.ev(X[:, 0], X[:, 1])
            predictions.append(pred_values)

        return np.column_stack(predictions)
