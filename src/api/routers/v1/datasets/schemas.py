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
    type: str
    name: str
    parameters: List[ParameterDefinition]


class GeneratorsDiscoveryResponse(BaseModel):
    generators: List[GeneratorSchema]


class DatasetMetadataSchema(BaseModel):
    n_samples: int
    n_train: int
    n_test: int
    split_ratio: float
    random_state: int
    created_at: str


class DatasetSummary(BaseModel):
    name: str
    n_features: int
    n_objectives: int
    metadata: DatasetMetadataSchema
    trained_engines_count: int


class EngineInfo(BaseModel):
    solver_type: str
    version: int
    created_at: datetime


class DatasetDetailResponse(BaseModel):
    name: str
    objectives_dim: int
    decisions_dim: int
    metadata: DatasetMetadataSchema
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
    split_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
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
