from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


class EstimatorDiagnosticRequest(BaseModel):
    """Identifies a specific diagnostic run to visualize."""

    type: EstimatorTypeEnum = Field(..., description="The type of estimator evaluated.")
    version: int = Field(..., description="The specific version of the model.")
    run_number: int | None = Field(
        default=None,
        description="The sequential run ID (1, 2...). If None, latest run is loaded.",
    )


class VisualizeDiagnosticsCommand(BaseModel):
    """Command payload for visualizing saved diagnostic results."""

    dataset_name: str = Field(
        ...,
        description="The dataset context the diagnostics were run on.",
        examples=["cocoex_f5"],
    )

    requests: list[EstimatorDiagnosticRequest] = Field(
        ...,
        description="List of specific evaluation runs to load and compare.",
        examples=[[{"type": "mdn", "version": 1, "run_number": 1}]],
    )
