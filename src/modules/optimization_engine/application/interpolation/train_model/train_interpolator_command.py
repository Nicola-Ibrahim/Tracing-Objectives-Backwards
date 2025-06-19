from pydantic import BaseModel, ConfigDict, Field

from ....domain.interpolation.entities.interpolator_type import InterpolatorType
from .dtos import InterpolatorParams


class TrainInterpolatorCommand(BaseModel):
    """
    Pydantic Command to encapsulate data required for training an interpolator model.
    It specifies the interpolator type and its parameters, not the instance itself.
    """

    data_source_name: str = Field(
        "pareto_data", description="The name of pareto data file"
    )

    type: InterpolatorType = Field(
        ...,
        description="The type of interpolator to be trained (e.g., NEURAL_NETWORK, GEODESIC).",
    )
    params: InterpolatorParams = Field(
        ...,
        description="Dictionary of parameters used to initialize/configure the interpolator instance.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
