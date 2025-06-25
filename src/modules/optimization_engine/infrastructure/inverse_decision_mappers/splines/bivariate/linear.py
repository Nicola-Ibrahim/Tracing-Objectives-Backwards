from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    Base1DInverseDecisionMapper,
)


class LinearInverseDecisionMapper(Base1DInverseDecisionMapper):
    _interp_func: Any | None = (
        None  # interp1d returns an object that acts like a function
    )

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        super().fit(objective_space_points, decision_space_points)
        if len(objective_space_points) < 2:
            raise ValueError(
                "Linear Inverse Decision Mapper requires at least 2 data points for fitting."
            )
        self._interp_func = interp1d(
            objective_space_points,
            decision_space_points,
            kind="linear",
            fill_value="extrapolate",
        )

    def predict(
        self, target_objective_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        if target_objective_points.ndim > 1:
            target_objective_points = target_objective_points.flatten()
        return self._interp_func(target_objective_points)
