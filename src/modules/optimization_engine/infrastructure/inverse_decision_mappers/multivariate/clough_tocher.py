import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator

from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)


class CloughTocherInverseDecisionMapper(BaseInverseDecisionMapper):
    def __init__(self) -> None:
        super().__init__()
        self._interp_func: CloughTocher2DInterpolator = None

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        # Call the parent's fit method for shared validation and dimension storage
        super().fit(objectives, decisions)

        # Perform specific validation for this interpolator
        if objectives.shape[1] != 2:
            raise ValueError(
                "CloughTocherInverseDecisionMapper requires 2D objective data."
            )
        if len(objectives) < 4:
            raise ValueError(
                "CloughTocherInverseDecisionMapper requires at least 4 data points for fitting."
            )

        # Perform specific fitting logic
        self._interp_func = CloughTocher2DInterpolator(
            points=objectives, values=decisions
        )

    def predict(
        self,
        target_objectives: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Perform validation specific to this method
        if self._interp_func is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if target_objectives.ndim == 1:
            target_objectives = target_objectives.reshape(-1, 1)

        if target_objectives.shape[1] != self._objective_dim:
            raise ValueError(
                f"Target objectives must have {self._objective_dim} dimensions, "
                f"but got {target_objectives.shape[1]} dimensions."
            )

        # Call the fitted interpolation function
        return self._interp_func(target_objectives)
