from typing import Literal

from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class VisualizeModelPerformanceCommand(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with the model.",
        examples=["dataset"],
    )
    data_file_name: str = Field(
        ...,
        description="Raw dataset file identifier (CLI-provided).",
        examples=["dataset"],
    )
    processed_file_name: str = Field(
        ...,
        description="Processed dataset file identifier (CLI-provided).",
        examples=["dataset"],
    )
    estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Name of the interpolator to load.",
        examples=[EstimatorTypeEnum.MDN.value],
    )
    mapping_direction: Literal["inverse", "forward"] = Field(
        ...,
        description="Which mapping direction ('inverse' or 'forward') to visualize.",
        examples=["inverse"],
    )
    model_number: int | None = Field(
        ...,
        ge=1,
        description="Nth most recent model to visualize (1 = latest).",
        examples=[1],
    )

    class Config:
        use_enum_values = True
