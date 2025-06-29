from pydantic import BaseModel, Field
from sklearn.gaussian_process.kernels import Kernel

from ....domain.interpolation.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)


class InverseDecisionMapperParams(BaseModel):
    pass


class CloughTocherInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: str = Field(
        InverseDecisionMapperType.CLOUGH_TOCHER_ND.value,
        description="Type of the Clough-Tocher interpolation method.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class NeuralNetworkInverserDecisionMapperParams(InverseDecisionMapperParams):
    type: str = Field(
        InverseDecisionMapperType.NEURAL_NETWORK_ND.value,
        description="Type of the neural network interpolation method.",
    )
    objective_dim: int = Field(2, description="The dimension of objective space")

    decision_dim: int = Field(2, description="The dimension of decision space")

    epochs: int = Field(
        1000, gt=0, description="Number of epochs for training the neural network."
    )
    learning_rate: float = Field(
        0.001, gt=0, description="Learning rate for the neural network optimizer."
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class GeodesicInterpolatorParams(InverseDecisionMapperParams):
    num_paths: int = Field(100, gt=0, description="Number of geodesic paths to sample.")
    max_iterations: int = Field(
        50, gt=0, description="Max iterations for path finding."
    )

    # Add other Geodesic specific parameters
    class Config:
        extra = "forbid"


class NearestNeighborInverseDecisoinMapperParams(InverseDecisionMapperParams):
    type: str = Field(
        InverseDecisionMapperType.NEAREST_NEIGHBORS_ND.value,
        description="Type of the nearest neighbor interpolation method.",
    )

    class Config:
        extra = "forbid"


class LinearInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: str = Field(
        InverseDecisionMapperType.LINEAR_ND.value,
        description="Type of the linear interpolation method.",
    )

    class Config:
        extra = "forbid"


class RBFInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: str = Field(
        InverseDecisionMapperType.RBF_ND.value,
        description="Type of the radial basis function interpolation method.",
    )
    n_neighbors: int = Field(
        10, gt=0, description="Number of nearest neighbors for RBF interpolation."
    )
    kernel: str = Field(
        "thin_plate_spline",
        description="Type of kernel to use for RBF interpolation.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class GaussianProcessInverseDecisionMapperParams(InverseDecisionMapperParams):
    """
    Pydantic model to define and validate parameters for a
    GaussianProcessInverseDecisionMapper.
    """

    type: str = Field(
        InverseDecisionMapperType.GAUSSIAN_PROCESS_ND.value,
        description="Type of the gaussian process interpolation method.",
    )

    kernel: Kernel | str = Field(
        "Matern",
        description="""The kernel (covariance function) to use for the Gaussian Process.
        Can be a string ('RBF' or 'Matern') or a scikit-learn Kernel object.""",
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
