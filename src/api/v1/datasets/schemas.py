from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    name: str
    type: str
    required: bool
    default: Any | None = None
    options: list[Any] | None = None
    description: str | None = None


class GeneratorSchema(BaseModel):
    id: str
    name: str
    parameters: List[ParameterDefinition]


class GeneratorsDiscoveryResponse(BaseModel):
    generators: List[GeneratorSchema]


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
    samples: int
    objectives_count: int
    decisions_count: int
    X: list[list[float]]
    y: list[list[float]]
    is_pareto: list[bool]
    bounds: dict[str, tuple[float, float]]
    trained_engines: list[EngineInfo] = Field(default_factory=list)


class DatasetGenerationRequest(BaseModel):
    dataset_name: str
    generator_type: str = "coco_pymoo"
    params: dict[str, Any] = Field(
        default_factory=dict, description="Generator-specific hyperparameters"
    )
    split_ratio: float = 0.2
    random_state: int = 42


class DatasetGenerationResponse(BaseModel):
    status: str
    name: str
    path: str


class DatasetDeleteResponse(BaseModel):
    name: str
    engines_removed: int = 0
    status: str


class BulkDeleteDatasetsRequest(BaseModel):
    dataset_names: List[str]
