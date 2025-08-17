import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator

from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)


class RBFInverseDecisionMapper(BaseInverseDecisionMapper):
    _interp_func: RBFInterpolator | None = None

    def __init__(
        self, n_neighbors: int = 10, kernel: str = "thin_plate_spline"
    ) -> None:
        """Initialize the RBF Inverse Decision Mapper.

        Args:
            n_neighbors (int): Number of nearest neighbors to consider for interpolation.
            kernel (str): Type of kernel to use for interpolation. Default is 'gaussian'.
                `linear` : -r
                `thin_plate_spline` : r**2 * log(r)
                `cubic` : r**3
                `quintic` : -r**5
                `multiquadric` : -sqrt(1 + r**2)
                `inverse_multiquadric` : 1/sqrt(1 + r**2)
                `inverse_quadratic` : 1/(1 + r**2)
                `gaussian` : exp(-r**2)


        Raises:
            ValueError: If neighbors is less than 1.
            ValueError: If kernel is not a valid RBF kernel type.

        """
        super().__init__()
        self._interp_func: RBFInterpolator = None
        self.neighbors = n_neighbors
        self.kernel = kernel

        # Validate kernel and neighbors during init
        valid_kernels = {
            "linear",
            "thin_plate_spline",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        }
        if kernel not in valid_kernels:
            raise ValueError(
                f"Invalid kernel '{kernel}'. Must be one of {valid_kernels}."
            )
        if n_neighbors < 1:
            raise ValueError("Neighbors must be at least 1.")

    def fit(
        self,
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> None:
        # 1. Call the parent's fit method for shared validation and dimension storage
        super().fit(objectives, decisions)

        # 2. Perform specific validation for this interpolator
        if len(objectives) < 1:
            raise ValueError(
                "RBF Inverse Decision Mapper requires at least 1 data point for fitting."
            )

        # 3. Perform specific fitting logic
        # RBFInterpolator expects `y` as the coordinates and `d` as the values.
        self._interp_func = RBFInterpolator(
            y=objectives,
            d=decisions,
            neighbors=self.neighbors,
            kernel=self.kernel,
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
