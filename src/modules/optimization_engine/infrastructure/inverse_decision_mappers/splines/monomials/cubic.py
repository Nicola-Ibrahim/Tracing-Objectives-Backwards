import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    Base1DInverseDecisionMapper,
)


class CubicSplineInverseDecisionMapper(Base1DInverseDecisionMapper):
    _interp_func: CubicSpline | None = None

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        super().fit(objective_space_points, decision_space_points)
        if len(objective_space_points) < 4:
            raise ValueError("CubicSpline requires at least 4 data points for fitting.")
        self._interp_func = CubicSpline(objective_space_points, decision_space_points)

    def predict(
        self, target_objective_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        if target_objective_points.ndim > 1:
            target_objective_points = target_objective_points.flatten()
        return self._interp_func(target_objective_points)
