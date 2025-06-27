# import numpy as np
# from numpy.typing import NDArray
# from scipy.interpolate import CubicSpline

# from ....domain.interpolation.interfaces.base_inverse_decision_mappers import (
#     BaseInverseDecisionMapper,
# )


# class CubicSplineInverseDecisionMapper(BaseInverseDecisionMapper):
#     _interp_func: CubicSpline | None = None

#     def fit(
#         self,
#         objectives: NDArray[np.float64],
#         decisions: NDArray[np.float64],
#     ) -> None:
#         super().fit(objectives, decisions)
#         if len(objectives) < 4:
#             raise ValueError("CubicSpline requires at least 4 data points for fitting.")

#         # Split the input arrays to ensure they are 1D and create two cubic spline functions
#         if objectives.ndim > 1:
#             objectives = objectives.flatten()

#         self._interp_func = CubicSpline(
#             x=objectives, y=decisions
#         )

#     def predict(
#         self, target_objectives: NDArray[np.float64]
#     ) -> NDArray[np.float64]:
#         if self._interp_func is None:
#             raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")
#         if target_objectives.ndim > 1:
#             target_objectives = target_objectives.flatten()
#         return self._interp_func(target_objectives)
