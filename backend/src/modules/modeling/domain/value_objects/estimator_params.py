from typing import Any

from pydantic import BaseModel, Field

from ..enums.metric_type import MetricTypeEnum
from ..enums.transform_type import TransformTypeEnum


class EstimatorParamsBase(BaseModel):
    pass


class TransformConfig(BaseModel):
    """
    Configuration for a pipeline generic transform step.
    """

    type: TransformTypeEnum = Field(
        ..., description="The type of the transform to use."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to the transform type.",
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
