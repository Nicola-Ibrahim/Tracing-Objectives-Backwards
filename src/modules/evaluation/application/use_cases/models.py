from pydantic import BaseModel, Field

from ....modeling.domain.enums.estimator_type import EstimatorTypeEnum


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


class InverseEstimatorDiagnosticCandidate(BaseModel):
    """Identifies a specific inverse estimator run to visualize."""

    type: EstimatorTypeEnum = Field(..., description="The type of estimator evaluated.")
    version: int = Field(..., description="The specific version of the model.")
    run_number: int | None = Field(
        default=None,
        description="The sequential run ID (1, 2...). If None, latest run is loaded.",
    )
