import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import SmoothBivariateSpline

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class SplineInverseDecisionMapper(BaseInverseDecisionMapper):
    """
    Inverse Decision Mapper using SciPy's SmoothBivariateSpline for 2D objective spaces.

    NOTE: This implementation is limited to objective spaces with 2 dimensions.
    """

    # We must store a list of spline functions for each output dimension.
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
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        # 1. Call the parent's fit method for universal validation
        super().fit(objectives, decisions)

        # 2. Perform specific validation for this interpolator
        if self._objective_dim != 2:
            raise ValueError(
                "SplineInverseDecisionMapper requires objectives with exactly 2 dimensions (x, y)."
            )

        # 3. Fit a separate spline for each output dimension.
        # This is necessary because SmoothBivariateSpline can only handle a single output (z).
        self._spline_funcs = []
        for i in range(self._decision_dim):
            # SmoothBivariateSpline expects 1D arrays for x, y, z
            spline_func = SmoothBivariateSpline(
                x=objectives[:, 0],
                y=objectives[:, 1],
                z=decisions[:, i],  # Pass only one decision dimension as the output
                s=self.s,
            )
            self._spline_funcs.append(spline_func)

    def predict(
        self,
        target_objectives: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._spline_funcs is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)

        if target_objectives.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {target_objectives.shape[1]} dimensions."
            )

        # Call each fitted spline and stack the results
        predictions = []
        for spline_func in self._spline_funcs:
            # `ev` (evaluate) method for bivariate splines
            pred_values = spline_func.ev(
                target_objectives[:, 0], target_objectives[:, 1]
            )
            predictions.append(pred_values)

        return np.column_stack(predictions)
