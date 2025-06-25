from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (
    PchipInterpolator,
)

from .....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    Base1DInverseDecisionMapper,
)


class PchipInverseDecisionMapper(Base1DInverseDecisionMapper):
    _interp_func: PchipInterpolator | None = None

    def fit(
        self,
        objective_space_points: NDArray[np.float64],
        decision_space_points: NDArray[np.float64],
    ) -> None:
        super().fit(
            objective_space_points, decision_space_points
        )  # Call parent validation for 1D arrays
        if len(objective_space_points) < 2:
            raise ValueError(
                "PchipInterpolator requires at least 2 data points for fitting."
            )
        self._interp_func = PchipInterpolator(
            objective_space_points, decision_space_points
        )

    def predict(
        self, target_objective_points: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # Ensure target_objective_points are flattened for 1D scipy interpolators if not already
        if target_objective_points.ndim > 1:
            target_objective_points = target_objective_points.flatten()
        return self._interp_func(target_objective_points)
