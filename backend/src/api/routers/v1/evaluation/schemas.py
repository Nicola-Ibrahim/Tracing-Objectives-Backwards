from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class EngineCandidateSchema(BaseModel):
    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(default=None, description="If None, latest is used")


class DiagnoseRequest(BaseModel):
    dataset_name: str
    candidates: List[EngineCandidateSchema]
    num_samples: int = 200
    scale_method: Literal["sd", "mad", "iqr"] = "sd"


class MetricSeries(BaseModel):
    x: List[float]
    y: List[float]


class DomainAssessmentData(BaseModel):
    """
    Structured data for a specific space (objective or decision).
    """

    ecdf: Dict[str, MetricSeries] = Field(default_factory=dict)
    calibration_curves: Dict[str, MetricSeries] = Field(default_factory=dict)
    # Metrics
    metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    warnings: List[str] = Field(default_factory=list)


class DiagnoseAsyncResponse(BaseModel):
    task_id: str
    status: str = "processing"


class PerformanceRequest(BaseModel):
    dataset_name: str
    engine: EngineCandidateSchema
    n_samples: int = 10


class PerformanceResponse(BaseModel):
    dataset_name: str
    solver_type: str
    version: int
    insights: dict[str, Any]
