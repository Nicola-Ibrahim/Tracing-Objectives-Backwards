from datetime import datetime

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    name: str
    n_samples: int
    n_features: int
    n_objectives: int
    trained_engines_count: int


class EngineInfo(BaseModel):
    solver_type: str
    version: int
    created_at: datetime


class DatasetDetailResponse(BaseModel):
    name: str
    X: list[list[float]]
    y: list[list[float]]
    is_pareto: list[bool]
    bounds: dict[str, tuple[float, float]]
    trained_engines: list[EngineInfo] = Field(default_factory=list)


class DatasetGenerationRequest(BaseModel):
    function_id: int
    population_size: int = 200
    n_var: int = 2
    generations: int = 20
    dataset_name: str


class DatasetGenerationResponse(BaseModel):
    status: str
    name: str
    path: str


class DatasetDeleteResponse(BaseModel):
    name: str
    engines_removed: int
