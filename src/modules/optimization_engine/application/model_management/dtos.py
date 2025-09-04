from typing import Any, Literal

from pydantic import BaseModel, Field

from ...domain.model_management.enums.estimator_type_enum import (
    EstimatorTypeEnum,
)


class EstimatorParams(BaseModel):
    pass


class CloughTocherEstimatorParams(EstimatorParams):
    type: str = Field(
        EstimatorTypeEnum.CLOUGH_TOCHER_ND.value,
        description="Type of the Clough-Tocher interpolation method.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class NeuralNetworkEstimatorParams(EstimatorParams):
    type: str = Field(
        EstimatorTypeEnum.NEURAL_NETWORK_ND.value,
        description="Type of the neural network interpolation method.",
    )
    objective_dim: int = Field(2, description="The dimension of objective space")

    decision_dim: int = Field(2, description="The dimension of decision space")

    epochs: int = Field(
        1000, gt=0, description="Number of epochs for training the neural network."
    )
    learning_rate: float = Field(
        1e-4, gt=0, description="Learning rate for the neural network optimizer."
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class GeodesicInterpolatorParams(EstimatorParams):
    num_paths: int = Field(100, gt=0, description="Number of geodesic paths to sample.")
    max_iterations: int = Field(
        50, gt=0, description="Max iterations for path finding."
    )

    class Config:
        extra = "forbid"


class NearestNeighborEstimatorParams(EstimatorParams):
    type: str = Field(
        EstimatorTypeEnum.NEAREST_NEIGHBORS_ND.value,
        description="Type of the nearest neighbor interpolation method.",
    )

    class Config:
        extra = "forbid"


class LinearEstimatorParams(EstimatorParams):
    type: str = Field(
        EstimatorTypeEnum.LINEAR_ND.value,
        description="Type of the linear interpolation method.",
    )

    class Config:
        extra = "forbid"


class RBFEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for an
    RBFEstimator.
    """

    type: str = Field(
        EstimatorTypeEnum.RBF_ND.value,
        description="Type of the radial basis function interpolation method.",
    )
    n_neighbors: int = Field(
        10, gt=0, description="Number of nearest neighbors for RBF interpolation."
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


class GaussianProcessEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for a
    GaussianProcessEstimator.
    """

    type: str = Field(
        EstimatorTypeEnum.GAUSSIAN_PROCESS_ND.value,
        description="Type of the gaussian process interpolation method.",
    )

    kernel: Literal["Matern", "RBF"] = Field(
        "Matern",
        description="""The kernel (covariance function) to use for the Gaussian Process.
        Must be one of 'Matern' or 'RBF'.""",
    )

    alpha: float = Field(
        1e-10,
        ge=0.0,  # Adds a validation constraint: value must be >= 0.0
        description="""Value added to the diagonal of the kernel matrix for numerical stability.
        Must be a non-negative float.""",
    )

    n_restarts_optimizer: int = Field(
        10,
        ge=0,  # Adds a validation constraint: value must be >= 0
        description="""Number of restarts of the optimizer to find the kernel's hyperparameters.
        Setting to 0 performs no optimization. Must be a non-negative integer.""",
    )

    random_state: int = Field(
        42,
        description="""Seed for the random number generator to ensure reproducibility of the training process.""",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        arbitrary_types_allowed = True


class SplineEstimatorParams(EstimatorParams):
    """Pydantic model for SmoothBivariateSpline mapper parameters."""

    type: str = Field(
        EstimatorTypeEnum.SPLINE_ND.value,
        description="Type of the spline interpolation method.",
    )

    # `s` is the smoothing factor. 0 means interpolation (passes through all points).
    # A positive value will produce a smoother curve.
    s: float = Field(
        0.0, ge=0.0, description="Positive smoothing factor. 0 for interpolation."
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class KrigingEstimatorParams(EstimatorParams):
    """Pydantic model for OrdinaryKriging mapper parameters."""

    type: str = Field(
        EstimatorTypeEnum.KRIGING_ND.value,
        description="Type of the Kriging interpolation method.",
    )

    variogram_model: Literal["linear", "gaussian", "spherical", "exponential"] = Field(
        "linear", description="Type of variogram model to use for Kriging."
    )
    # The number of data points to use in a local neighborhood search
    # around the interpolation point. Use `None` to use all points.
    n_neighbors: int | None = Field(
        12,
        ge=1,
        description="Number of nearest neighbors to use for Kriging. None for all points.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class SVREstimatorParams(EstimatorParams):
    """Pydantic model for SVR mapper parameters."""

    type: str = Field(
        EstimatorTypeEnum.SVR_ND.value,
        description="Type of the SVR interpolation method.",
    )

    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field(
        "rbf", description="Specifies the kernel type to be used in the algorithm."
    )
    C: float = Field(
        1.0,
        gt=0.0,
        description="Regularization parameter. The strength of the regularization is inversely proportional to C.",
    )
    epsilon: float = Field(
        0.1,
        ge=0.0,
        description="Epsilon in the epsilon-SVR model. Specifies the epsilon-tube within which no penalty is associated with errors.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class MDNEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for an
    MDNEstimator.
    """

    type: str = Field(
        EstimatorTypeEnum.MDN_ND.value,
        description="Type of the Mixture Density Network interpolation method.",
    )
    num_mixtures: int = Field(
        5, gt=0, description="The number of Gaussian mixture components for the MDN."
    )
    # epochs: int = Field(500, gt=0, description="Number of epochs for training the MDN.")
    learning_rate: float = Field(
        1e-3, gt=0, description="Learning rate for the Adam optimizer."
    )

    class Config:
        extra = "forbid"


class CVAEEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for a
    CVAEEstimator.
    """

    type: str = Field(
        EstimatorTypeEnum.CVAE_ND.value,
        description="Type of the Conditional Variational Autoencoder interpolation method.",
    )
    latent_dim: int = Field(
        8, gt=0, description="Dimensionality of the latent space in the CVAE."
    )
    # epochs: int = Field(
    #     500, gt=0, description="Number of epochs for training the CVAE."
    # )
    learning_rate: float = Field(
        1e-4, gt=0, description="Learning rate for the Adam optimizer."
    )
    # device: Literal["cpu", "cuda"] = Field(
    #     "cpu", description="The device to run the CVAE model on ('cpu' or 'cuda')."
    # )

    class Config:
        extra = "forbid"


class NormalizerConfig(BaseModel):
    """
    Configuration for a normalizer.
    """

    type: Literal[
        "MinMaxScaler",
        "HypercubeNormalizer",
        "StandardNormalizer",
        "UnitVectorNormalizer",
        "LogNormalizer",
    ] = Field(..., description="The type of the normalizer to use.")
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the normalizer type."
    )


class ValidationMetricConfig(BaseModel):
    """
    Configuration for a validation metric.
    """

    type: str = Field(..., description="The type of the metric to use.")
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the metric type."
    )
