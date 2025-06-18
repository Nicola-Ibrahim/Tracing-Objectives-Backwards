from pydantic import BaseModel, Field


class InterpolatorParams(BaseModel):
    # Common parameters if any, or just a base for polymorphism
    pass


class NeuralNetworkInterpolatorParams(InterpolatorParams):
    hidden_layer_sizes: tuple[int, ...] = Field(
        ..., description="Sizes of the hidden layers."
    )
    activation: str = Field(
        "relu", description="Activation function for hidden layers."
    )
    solver: str = Field("adam", description="The solver for weight optimization.")
    learning_rate_init: float = Field(0.001, gt=0, description="Initial learning rate.")
    max_iter: int = Field(200, gt=0, description="Maximum number of iterations.")

    # Add other NN specific parameters as needed
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


class KNearestNeighborInterpolatorParams(InterpolatorParams):
    n_neighbors: int = Field(5, gt=0, description="Number of neighbors to use.")
    weights: str = Field(
        "uniform",
        description="Weight function used in prediction. 'uniform' or 'distance'.",
    )
    metric: str = Field("euclidean", description="Distance metric to use.")

    # Add other KNN specific parameters
    class Config:
        extra = "forbid"


class LinearInterpolatorParams(InterpolatorParams):
    # Linear might have fewer parameters, or none, just keep it for consistency
    pass

    class Config:
        extra = "forbid"
