import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator

from .....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    Base1DInverseDecisionMapper,
)


class RBFInverseDecisionMapper(
    Base1DInverseDecisionMapper
):  # RBF can also handle ND input, but often used for 1D problems as well.
    _interp_func: RBFInterpolator | None = None

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        # RBFInterpolator internally expects 2D input for x even if it's a 1D problem (n_samples, 1)
        x_prepared_for_rbf = objective_space_points.reshape(-1, 1)
        # Call parent fit for general validation, but pass the 1D original for its check
        # Then use the reshaped version for RBF specific logic.
        super(Base1DInverseDecisionMapper, self).fit(
            objective_space_points, decision_space_points
        )  # Call BaseInverseDecisionMapper directly for length checks
        if len(objective_space_points) < 1:
            raise ValueError(
                "RBF Inverse Decision Mapper requires at least 1 data point for fitting."
            )
        self._interp_func = RBFInterpolator(x_prepared_for_rbf, decision_space_points)

    def predict(
        self, target_objective_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # RBFInterpolator expects 2D input for evaluation points (n_samples, n_features)
        points_to_evaluate_prepared = target_objective_points.reshape(-1, 1)
        return self._interp_func(points_to_evaluate_prepared)
