from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.interpolate import RBFInterpolator

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum
from .....modeling.domain.interfaces.base_estimator import (
    DeterministicEstimator,
)
from .....modeling.domain.value_objects.estimator_params import EstimatorParamsBase


class RBFEstimatorParams(EstimatorParamsBase):
    """
    Pydantic model to define and validate parameters for an
    RBFEstimator.
    """

    type: Literal["rbf"] = Field(
        EstimatorTypeEnum.RBF.value,
        description="Type of the radial basis function interpolation method.",
    )
    n_neighbors: int = Field(
        10, gt=0, description="Number of nearest neighbors for RBF interpolation."
    )
    epsilon: float | None = Field(
        None,
        ge=0,
        description=(
            "Shape parameter that scales inputs to the RBF. If None, "
            "it is estimated using the average distance between nodes."
        ),
    )

    kernel: Literal[
        "linear",
        "thin_plate_spline",
        "cubic",
        "quintic",
        "multiquadric",
        "inverse_multiquadric",
        "inverse_quadratic",
        "gaussian",
    ] = Field(
        "thin_plate_spline",
        description="""Type of kernel to use for RBF interpolation.
        Options correspond to different basis functions:
        `linear` : -r
        `thin_plate_spline` : r**2 * log(r)
        `cubic` : r**3
        `quintic` : -r**5
        `multiquadric` : -sqrt(1 + r**2)
        `inverse_multiquadric` : 1/sqrt(1 + r**2)
        `inverse_quadratic` : 1/(1 + r**2)
        `gaussian` : exp(-r**2)
        """,
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        use_enum_values = True


class RBFEstimator(DeterministicEstimator):
    def __init__(self, params: RBFEstimatorParams) -> None:
        """Initialize the RBF Inverse Decision Mapper."""
        kernel = params.kernel
        n_neighbors = params.n_neighbors
        epsilon = params.epsilon
        super().__init__()
        self.params = params
        self._model: RBFInterpolator = None
        self.neighbors = n_neighbors
        self.kernel = kernel
        self.epsilon = epsilon

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

    @property
    def type(self) -> str:
        return EstimatorTypeEnum.RBF.value

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        super().fit(X, y)

        if len(X) < 1:
            raise ValueError(
                "RBF Mapper requires at least 1 data point for fitting."
            )

        # Combine X and y to find unique rows
        combined_data = np.hstack((X, y))
        unique_data, unique_indices = np.unique(
            combined_data, axis=0, return_index=True
        )

        # Filter X and y to keep only unique data points
        X_unique = X[unique_indices]
        y_unique = y[unique_indices]

        # Check if singular matrix might be issue with small number of points
        if len(X_unique) < 2 and self.kernel in [
            "thin_plate_spline",
            "cubic",
            "quintic",
        ]:
            raise ValueError(
                f"Kernel '{self.kernel}' requires at least 2 unique data points."
            )

        self._model = RBFInterpolator(
            y=X_unique,
            d=y_unique,
            neighbors=self.neighbors,
            kernel=self.kernel,
            epsilon=self.epsilon,
        )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("Mapper has not been fitted yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._X_dim:
            raise ValueError(
                f"Target objectives must have {self._X_dim} dimensions, "
                f"but got {X.shape[1]} dimensions."
            )

        return self._model(X)

    def to_checkpoint(self) -> dict:
        """
        Serialize model state to a checkpoint dictionary.
        Note: Currently RBFInterpolator is also picked in repository logic.
        """
        if self._model is None:
            raise RuntimeError("Estimator not fitted.")

        return {
            "params": self.params.model_dump(),
            "X_dim": self._X_dim,
            "y_dim": self._y_dim,
            # We don't serialize the RBFInterpolator to JSON/TOML easily,
            # but this satisfies the abstract interface.
            # GenerationContext uses pickle for the actual persistence.
        }

    @classmethod
    def from_checkpoint(cls, parameters: dict) -> "RBFEstimator":
        """Reconstruct from checkpoint."""
        params = RBFEstimatorParams(**parameters["params"])
        instance = cls(params)
        instance._X_dim = parameters.get("X_dim")
        instance._y_dim = parameters.get("y_dim")
        return instance
