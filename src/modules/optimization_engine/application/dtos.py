from typing import Any, Literal

from pydantic import BaseModel, Field

from ..domain.modeling.enums.estimator_type import (
    EstimatorTypeEnum,
)
from ..domain.modeling.enums.metric_type import MetricTypeEnum
from ..domain.modeling.enums.normalizer_type import NormalizerTypeEnum


class EstimatorParams(BaseModel):
    pass


class COCOEstimatorParams(EstimatorParams):
    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.COCO.value,
        description="Type of the COCO interpolation method.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        use_enum_values = True


class CloughTocherEstimatorParams(EstimatorParams):
    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.CLOUGH_TOCHER_ND.value,
        description="Type of the Clough-Tocher interpolation method.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        use_enum_values = True


class NeuralNetworkEstimatorParams(EstimatorParams):
    type: EstimatorTypeEnum = Field(
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
        use_enum_values = True


class GeodesicInterpolatorParams(EstimatorParams):
    num_paths: int = Field(100, gt=0, description="Number of geodesic paths to sample.")
    max_iterations: int = Field(
        50, gt=0, description="Max iterations for path finding."
    )

    class Config:
        extra = "forbid"


class RBFEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for an
    RBFEstimator.
    """

    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.RBF.value,
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
        use_enum_values = True


class GaussianProcessEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for a
    GaussianProcessEstimator.
    """

    type: EstimatorTypeEnum = Field(
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
        use_enum_values = True


class MDNEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for an
    MDNEstimator.
    """

    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.MDN.value,
        description="Type of the Mixture Density Network interpolation method.",
    )
    num_mixtures: int = Field(
        10, gt=0, description="The number of Gaussian mixture components for the MDN."
    )
    learning_rate: float = Field(
        1e-3, gt=0, description="Learning rate for the Adam optimizer."
    )
    epochs: int = Field(100, gt=0, description="Number of training epochs.")
    batch_size: int = Field(128, gt=0, description="Mini-batch size used in training.")
    hidden_layers: list[int] = Field(
        [256, 128, 128],
        description="List defining the number of units in each hidden layer of the MDN.",
    )
    gmm_boost: bool = Field(
        False, description="Whether to apply GMM boosting to the MDN."
    )
    val_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Validation split fraction used during training.",
    )
    weight_decay: float = Field(
        1e-4, ge=0.0, description="L2 weight decay applied during optimization."
    )
    clip_grad_norm: float | None = Field(
        None,
        ge=0.0,
        description="Optional gradient norm clipping threshold.",
    )
    seed: int = Field(42, description="Random seed for reproducible MDN training.")

    class Config:
        extra = "forbid"
        use_enum_values = True


class CVAEEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for a
    CVAEEstimator.
    """

    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.CVAE.value,
        description="Type of the Conditional Variational Autoencoder interpolation method.",
    )
    latent_dim: int = Field(
        8, gt=0, description="Dimensionality of the latent space in the CVAE."
    )
    learning_rate: float = Field(
        1e-4, gt=0, description="Learning rate for the Adam optimizer."
    )
    beta: float = Field(0.1, ge=0.0, description="Final KL divergence weight.")
    kl_warmup: int = Field(
        100,
        ge=0,
        description="Number of epochs used to warm-up the KL weight from 0 to beta.",
    )
    free_nats: float = Field(
        0.0,
        ge=0.0,
        description="Free nats threshold applied to KL divergence per dimension.",
    )
    epochs: int = Field(100, gt=0, description="Number of training epochs.")
    batch_size: int = Field(128, gt=0, description="Mini-batch size used in training.")
    val_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Validation split fraction used during training.",
    )
    random_state: int = Field(42, description="Random seed for data splitting.")

    class Config:
        extra = "forbid"
        use_enum_values = True


class CVAEMDNEstimatorParams(EstimatorParams):
    """
    Pydantic model to define and validate parameters for a
    CVAEEstimator.
    """

    type: EstimatorTypeEnum = Field(
        EstimatorTypeEnum.CVAE_MDN.value,
        description="Type of the Conditional Variational Autoencoder interpolation method.",
    )
    latent_dim: int = Field(
        8, gt=0, description="Dimensionality of the latent space in the CVAE."
    )
    learning_rate: float = Field(
        1e-4, gt=0, description="Learning rate for the Adam optimizer."
    )
    n_components: int = Field(5, gt=0, description="Number of MDN components")
    beta: float = Field(0.1, ge=0.0, description="Final KL divergence weight.")
    kl_warmup: int = Field(
        100,
        ge=0,
        description="Number of epochs used to warm-up the KL weight from 0 to beta.",
    )
    free_nats: float = Field(
        0.0,
        ge=0.0,
        description="Free nats threshold applied to KL divergence per dimension.",
    )
    epochs: int = Field(100, gt=0, description="Number of training epochs.")
    batch_size: int = Field(128, gt=0, description="Mini-batch size used in training.")
    val_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Validation split fraction used during training.",
    )
    random_state: int = Field(42, description="Random seed for data splitting.")

    class Config:
        extra = "forbid"
        use_enum_values = True


class NormalizerConfig(BaseModel):
    """
    Configuration for a normalizer.
    """

    type: NormalizerTypeEnum = Field(
        ..., description="The type of the normalizer to use."
    )
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the normalizer type."
    )

    class Config:
        use_enum_values = True


class ValidationMetricConfig(BaseModel):
    """
    Configuration for a validation metric.
    """

    type: MetricTypeEnum = Field(..., description="The type of the metric to use.")
    params: dict[str, Any] = Field(
        {}, description="Parameters specific to the metric type."
    )

    class Config:
        use_enum_values = True
