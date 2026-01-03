from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from ..enums.estimator_type import EstimatorTypeEnum
from ..enums.metric_type import MetricTypeEnum
from ..enums.normalizer_type import NormalizerTypeEnum


class EstimatorParamsBase(BaseModel):
    pass


class ActivationFunctionEnum(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"


class DistributionFamilyEnum(Enum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    LOGNORMAL = "lognormal"


class OptimizerFunctionEnum(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"


class INNOptimizerEnum(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class COCOEstimatorParams(EstimatorParamsBase):
    type: Literal["coco"] = Field(
        EstimatorTypeEnum.COCO.value,
        description="Type of the COCO interpolation method.",
    )
    problem_name: str = Field(
        "bbob-biobj",
        description="COCO suite name for bi-objective problems.",
    )
    function_indices: int = Field(
        5,
        ge=1,
        le=55,
        description="COCO function index (1-55 for bbob-biobj).",
    )
    instance_indices: int = Field(1, ge=1, description="COCO instance index.")
    dimensions: int = Field(2, ge=1, description="Problem dimensionality.")

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        use_enum_values = True


class NeuralNetworkEstimatorParams(EstimatorParamsBase):
    type: Literal["neural_network_nd"] = Field(
        EstimatorTypeEnum.NEURAL_NETWORK_ND.value,
        description="Type of the neural network interpolation method.",
    )
    objective_dim: int = Field(2, description="The dimension of objective space")

    decision_dim: int = Field(2, description="The dimension of decision space")

    epochs: int = Field(
        1000, gt=0, description="Number of epochs for training the neural network."
    )
    learning_rate: float = Field(
        1e-3, gt=0, description="Learning rate for the neural network optimizer."
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
        use_enum_values = True


class NearestNeighborsEstimatorParams(EstimatorParamsBase):
    type: Literal["nearest_neighbors_nd"] = Field(
        EstimatorTypeEnum.NEAREST_NEIGHBORS_ND.value,
        description="Type of the nearest-neighbors interpolation method.",
    )

    class Config:
        extra = "forbid"
        use_enum_values = True


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


class GaussianProcessEstimatorParams(EstimatorParamsBase):
    """
    Pydantic model to define and validate parameters for a
    GaussianProcessEstimator.
    """

    type: Literal["gaussian_process_nd"] = Field(
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


class MDNEstimatorParams(EstimatorParamsBase):
    """
    Pydantic model to define and validate parameters for an
    MDNEstimator.
    """

    type: Literal["mdn"] = Field(
        EstimatorTypeEnum.MDN.value,
        description="Type of the Mixture Density Network interpolation method.",
    )
    num_mixtures: int = Field(
        -1,
        description="Number of mixture components (-1 triggers BIC-based selection).",
    )
    learning_rate: float = Field(
        1e-4, gt=0, description="Learning rate for the optimizer."
    )
    distribution_family: DistributionFamilyEnum = Field(
        DistributionFamilyEnum.NORMAL,
        description="Distribution family used for mixture components.",
    )
    gmm_boost: bool = Field(
        False, description="Whether to apply GMM-based initialization."
    )
    hidden_layers: list[int] = Field(
        [256, 128, 128],
        description="List defining the number of units in each hidden layer of the MDN.",
    )
    hidden_activation_fn_name: ActivationFunctionEnum = Field(
        ActivationFunctionEnum.RELU,
        description="Activation function for hidden layers.",
    )
    optimizer_fn_name: OptimizerFunctionEnum = Field(
        OptimizerFunctionEnum.ADAM,
        description="Optimizer used for training.",
    )
    verbose: bool = Field(False, description="Enable verbose training logs.")
    epochs: int = Field(100, gt=0, description="Number of training epochs.")
    batch_size: int = Field(32, gt=0, description="Mini-batch size used in training.")
    val_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Validation split fraction used during training.",
    )
    weight_decay: float = Field(
        0.0, ge=0.0, description="L2 weight decay applied during optimization."
    )
    clip_grad_norm: float | None = Field(
        None,
        ge=0.0,
        description="Optional gradient norm clipping threshold.",
    )
    seed: int = Field(44, description="Random seed for reproducible MDN training.")

    class Config:
        extra = "forbid"
        use_enum_values = True


class CVAEEstimatorParams(EstimatorParamsBase):
    """
    Pydantic model to define and validate parameters for a
    CVAEEstimator.
    """

    type: Literal["cvae"] = Field(
        EstimatorTypeEnum.CVAE.value,
        description="Type of the Conditional Variational Autoencoder interpolation method.",
    )
    latent_dim: int = Field(
        8, gt=0, description="Dimensionality of the latent space in the CVAE."
    )
    learning_rate: float = Field(
        0.001, gt=0, description="Learning rate for the Adam optimizer."
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
    hidden: int = Field(128, gt=0, description="Hidden layer width for CVAE nets.")
    decoder_min_logvar: float = Field(
        -6.0, description="Lower clamp for decoder log-variance."
    )
    decoder_max_logvar: float = Field(
        4.0, description="Upper clamp for decoder log-variance."
    )
    prior_min_logvar: float = Field(
        -4.0, description="Lower clamp for prior log-variance."
    )
    prior_max_logvar: float = Field(
        2.0, description="Upper clamp for prior log-variance."
    )
    epochs: int = Field(200, gt=0, description="Number of training epochs.")
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


class INNEstimatorParams(EstimatorParamsBase):
    type: Literal["inn"] = Field(
        EstimatorTypeEnum.INN.value,
        description="Type of the invertible neural network estimator.",
    )
    num_coupling_layers: int = Field(
        6, ge=1, description="Number of coupling transformations."
    )
    hidden_dim: int = Field(
        128, gt=0, description="Hidden layer size in coupling networks."
    )
    use_batch_norm: bool = Field(
        True, description="Use batch normalization between layers."
    )
    learning_rate: float = Field(
        1e-3, gt=0, description="Initial learning rate."
    )
    optimizer_name: INNOptimizerEnum = Field(
        INNOptimizerEnum.ADAM, description="Optimizer choice."
    )
    weight_decay: float = Field(
        1e-5, ge=0, description="L2 regularization strength."
    )
    epochs: int = Field(100, gt=0, description="Maximum training epochs.")
    batch_size: int = Field(128, gt=0, description="Batch size for training.")
    val_size: float = Field(
        0.2,
        gt=0.0,
        lt=1.0,
        description="Validation set fraction.",
    )
    clip_grad_norm: float | None = Field(
        5.0, ge=0.0, description="Gradient clipping threshold."
    )
    lr_scheduler: bool = Field(
        True, description="Use learning rate scheduler."
    )
    lr_decay_factor: float = Field(
        0.5, gt=0.0, description="LR reduction factor for scheduler."
    )
    lr_patience: int = Field(
        10, ge=1, description="Patience for LR scheduler."
    )
    early_stopping_patience: int = Field(
        20, ge=1, description="Patience for early stopping."
    )
    verbose: bool = Field(True, description="Print training progress.")
    seed: int = Field(42, description="Random seed for reproducibility.")

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
        default_factory=dict,
        description="Parameters specific to the normalizer type.",
    )

    class Config:
        use_enum_values = True


class ValidationMetricConfig(BaseModel):
    """
    Configuration for a validation metric.
    """

    type: MetricTypeEnum = Field(..., description="The type of the metric to use.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to the metric type.",
    )

    class Config:
        use_enum_values = True


EstimatorParams = Annotated[
    Union[
        COCOEstimatorParams,
        NeuralNetworkEstimatorParams,
        NearestNeighborsEstimatorParams,
        RBFEstimatorParams,
        GaussianProcessEstimatorParams,
        MDNEstimatorParams,
        CVAEEstimatorParams,
        INNEstimatorParams,
    ],
    Field(discriminator="type"),
]
