from pydantic import BaseModel, Field

from ....domain.interpolation.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)


class InverseDecisionMapperParams(BaseModel):
    pass


class CloughTocherInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: InverseDecisionMapperType = Field(
        InverseDecisionMapperType.CLOUGH_TOCHER_ND.value,
        description="Type of the Clough-Tocher interpolation method.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class NeuralNetworkInverserDecisionMapperParams(InverseDecisionMapperParams):
    type: InverseDecisionMapperType = Field(
        InverseDecisionMapperType.NEURAL_NETWORK_ND.value,
        description="Type of the neural network interpolation method.",
    )
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
    type: InverseDecisionMapperType = Field(
        InverseDecisionMapperType.NEAREST_NEIGHBORS_ND.value,
        description="Type of the nearest neighbor interpolation method.",
    )

    class Config:
        extra = "forbid"


class LinearInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: InverseDecisionMapperType = Field(
        InverseDecisionMapperType.LINEAR_ND.value,
        description="Type of the linear interpolation method.",
    )

    class Config:
        extra = "forbid"


class RBFInverseDecisionMapperParams(InverseDecisionMapperParams):
    type: InverseDecisionMapperType = Field(
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
