from typing import Any

from pydantic import BaseModel, Field

from ..enums.metric_type import MetricTypeEnum
from ..enums.normalizer_type import NormalizerTypeEnum


class EstimatorParamsBase(BaseModel):
    pass


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
