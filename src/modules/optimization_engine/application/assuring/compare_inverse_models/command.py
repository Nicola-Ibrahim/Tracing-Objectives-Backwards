from pydantic import BaseModel, Field

from ....domain.modeling.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type and optional version)."""

    type: EstimatorTypeEnum = Field(
        ...,
        examples=[EstimatorTypeEnum.MDN.value],
    )
    version: int | None = Field(
        ...,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
        examples=[1],
    )


class CompareInverseModelsCommand(BaseModel):
    """Command payload for comparing inverse model candidates on a single dataset."""

    dataset_name: str = Field(
        ...,
        description="Dataset identifier to use for comparison.",
        examples=["dataset"],
    )

    candidates: list[InverseEstimatorCandidate] = Field(
        ...,
        description="List of model candidates to compare.",
        examples=[[{"type": EstimatorTypeEnum.MDN.value, "version": 1}]],
    )

    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Type of the forward estimator (simulator) to use for validation.",
        examples=[EstimatorTypeEnum.MDN.value],
    )

    num_samples: int = Field(
        ...,
        description="Number of samples to draw from the inverse model for each target.",
        examples=[250],
    )
