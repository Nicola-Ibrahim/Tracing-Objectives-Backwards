import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

from .....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseNDInverseDecisionMapper,
)


class LinearNDInverseDecisionMapper(BaseNDInverseDecisionMapper):
    _interp_func: LinearNDInterpolator | None = None

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        # Ensure objective_space_points is 2D (n_samples, n_features)
        if objective_space_points.ndim == 1:
            objective_space_points = objective_space_points.reshape(-1, 1)
        super().fit(
            objective_space_points, decision_space_points
        )  # Call parent validation
        if len(objective_space_points) < 3:
            raise ValueError(
                "LinearNDInverseDecisionMapper requires at least 3 data points for fitting."
            )
        self._interp_func = LinearNDInterpolator(
            objective_space_points, decision_space_points
        )

    def predict(
        self, target_objective_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # Ensure target_objective_points is 2D (n_samples, n_features) for ND interpolators
        if target_objective_points.ndim == 1:
            target_objective_points = target_objective_points.reshape(-1, 1)
        return self._interp_func(target_objective_points)
