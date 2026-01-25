from pydantic import BaseModel, Field

from ....domain.enums.estimator_type import EstimatorTypeEnum


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


class GenerateCandidatesCommand(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to use for comparison.",
        examples=["dataset"],
    )

    inverse_estimator: InverseEstimatorCandidate = Field(
        ...,
        description="inverse model candidate (type + version) to use for generation",
        examples={
            "type": EstimatorTypeEnum.MDN.value,
            "version": 1,
        },
    )
    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Name of the forward model to use for verification",
        examples=[EstimatorTypeEnum.MDN.value],
    )
    target_objective: list[float] = Field(
        ...,
        description="Target point in objective space (y1, y2, ...)",
        examples=[[0.5, 0.8]],
    )
    distance_tolerance: float = Field(
        ...,
        description="Distance tolerance for selecting candidates.",
        examples=[0.02],
    )
    n_samples: int = Field(
        ...,
        description="Number of samples to draw per inverse estimator.",
        examples=[250],
    )
