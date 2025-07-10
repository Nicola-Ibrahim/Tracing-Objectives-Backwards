import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import NearestNDInterpolator

from ....domain.interpolation.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)


class NearestNDInverseDecisionMapper(BaseInverseDecisionMapper):
    _interp_func: NearestNDInterpolator | None = None

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        # Ensure objectives is 2D (n_samples, n_features)
        if objectives.ndim == 1:
            objectives = objectives.reshape(-1, 1)
        super().fit(objectives, decisions)  # Call parent validation
        if len(objectives) < 1:
            raise ValueError(
                "NearestNDInverseDecisionMapper requires at least 1 data point for fitting."
            )
        self._interp_func = NearestNDInterpolator(x=objectives, y=decisions)

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # Ensure target_objectives is 2D (n_samples, n_features) for ND interpolators
        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)
        return self._interp_func(target_objectives)
