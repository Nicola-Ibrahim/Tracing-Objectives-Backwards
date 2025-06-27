from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class QuadraticInverseDecisionMapper(BaseInverseDecisionMapper):
    _interp_func: Any | None = None

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        super().fit(objectives, decisions)
        if len(objectives) < 3:
            raise ValueError(
                "Quadratic Inverse Decision Mapper requires at least 3 data points for fitting."
            )
        self._interp_func = interp1d(
            objectives,
            decisions,
            kind="quadratic",
            fill_value="extrapolate",
        )

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        if target_objectives.ndim > 1:
            target_objectives = target_objectives.flatten()
        return self._interp_func(target_objectives)
