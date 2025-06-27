import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import (
    PchipInterpolator,
)

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class PchipInverseDecisionMapper(BaseInverseDecisionMapper):
    _interp_func: PchipInterpolator | None = None

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        super().fit(objectives, decisions)  # Call parent validation for 1D arrays
        if len(objectives) < 2:
            raise ValueError(
                "PchipInterpolator requires at least 2 data points for fitting."
            )
        self._interp_func = PchipInterpolator(objectives, decisions)

    def predict(self, target_objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
        # Ensure target_objectives are flattened for 1D scipy interpolators if not already
        if target_objectives.ndim > 1:
            target_objectives = target_objectives.flatten()
        return self._interp_func(target_objectives)
