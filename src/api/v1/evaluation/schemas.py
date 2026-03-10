from typing import Any, List, Literal

from pydantic import BaseModel, Field


class EngineCandidateSchema(BaseModel):
    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(default=None, description="If None, latest is used")


class DiagnoseRequest(BaseModel):
    dataset_name: str
    candidates: List[EngineCandidateSchema]
    num_samples: int = 200
    scale_method: Literal["sd", "mad", "iqr"] = "sd"


class DiagnoseResponse(BaseModel):
    dataset_name: str
    engines: List[str]
    ecdf: dict[str, Any]
    pit: dict[str, Any]
    mace: dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


class PerformanceRequest(BaseModel):
    dataset_name: str
    engine: EngineCandidateSchema
    n_samples: int = 10


class PerformanceResponse(BaseModel):
    dataset_name: str
    solver_type: str
    version: int
    insights: dict[str, Any]
