from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type and optional version)."""

    type: EstimatorTypeEnum = Field(
        ...,
        examples=[EstimatorTypeEnum.MDN.value],
    )
    version: int = Field(
        default=1,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
        examples=[1],
    )


class CheckModelPerformanceCommand(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier associated with the model.",
        examples=["dataset"],
    )
    estimator: InverseEstimatorCandidate = Field(
        ...,
        description="Estimator type and optional version number (e.g., 1). If None, latest is used.",
        examples=[InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=1)],
    )
    n_samples: int = Field(
        default=2,
        ge=1,
        description="Number of samples to generate for visualization.",
        examples=[50],
    )

    class Config:
        use_enum_values = True
