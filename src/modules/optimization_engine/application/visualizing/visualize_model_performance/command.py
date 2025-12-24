from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class VisualizeModelPerformanceCommand(BaseModel):
    dataset_name: str | None = None
    data_file_name: str = "dataset"
    processed_file_name: str = "dataset"
    estimator_type: EstimatorTypeEnum = Field(
        ..., description="Name of the interpolator to load"
    )
    mapping_direction: Literal["inverse", "forward"] = Field(
        "inverse",
        description="Which mapping direction ('inverse' or 'forward') to visualize.",
    )
    model_number: int | None = Field(
        None,
        ge=1,
        description="Nth most recent model to visualize (1 = latest). Defaults to latest.",
    )

    @model_validator(mode="after")
    def _set_dataset_name(self) -> "VisualizeModelPerformanceCommand":
        if self.dataset_name is None:
            self.dataset_name = self.processed_file_name or self.data_file_name
        return self

    class Config:
        use_enum_values = True
