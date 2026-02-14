from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Identifies a specific inverse estimator run to visualize."""

    type: EstimatorTypeEnum = Field(..., description="The type of estimator evaluated.")
    version: int = Field(..., description="The specific version of the model.")
    run_number: int | None = Field(
        default=None,
        description="The sequential run ID (1, 2...). If None, latest run is loaded.",
    )


class VisualizeInverseEstimatorDiagnosticCommand(BaseModel):
    """Command payload for visualizing inverse estimator diagnostic results."""

    dataset_name: str = Field(
        ...,
        description="The dataset context the diagnostics were run on.",
        examples=["cocoex_f5"],
    )

    inverse_estimator_candidates: list[InverseEstimatorCandidate] = Field(
        ...,
        description="List of specific inverse estimator runs to load and compare.",
        examples=[
            [
                {
                    "type": "mdn",
                    "version": 1,
                    "run_number": 1,
                }
            ]
        ],
    )
