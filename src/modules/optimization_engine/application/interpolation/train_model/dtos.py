from pydantic import BaseModel, Field


class InterpolatorParams(BaseModel):
    # Common parameters if any, or just a base for polymorphism
    pass


class CloughTocherInterpolatorParams(InterpolatorParams):
    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class NeuralNetworkInterpolatorParams(InterpolatorParams):
    epochs: int = Field(
        1000, gt=0, description="Number of epochs for training the neural network."
    )
    learning_rate: float = Field(
        0.001, gt=0, description="Learning rate for the neural network optimizer."
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined


class GeodesicInterpolatorParams(InterpolatorParams):
    num_paths: int = Field(100, gt=0, description="Number of geodesic paths to sample.")
    max_iterations: int = Field(
        50, gt=0, description="Max iterations for path finding."
    )

    # Add other Geodesic specific parameters
    class Config:
        extra = "forbid"


class NearestNeighborInterpolatorParams(InterpolatorParams):
    class Config:
        extra = "forbid"


class LinearInterpolatorParams(InterpolatorParams):
    # Linear might have fewer parameters, or none, just keep it for consistency
    pass

    class Config:
        extra = "forbid"


class RBFInterpolatorParams(InterpolatorParams):
    n_neighbors: int = Field(
        10, gt=0, description="Number of nearest neighbors for RBF interpolation."
    )
    kernel: str = Field(
        "thin_plate_spline",
        description="Type of kernel to use for RBF interpolation.",
    )

    class Config:
        extra = "forbid"  # Forbid extra fields not defined
