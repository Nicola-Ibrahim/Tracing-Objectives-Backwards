from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    name: str
    type: str
    required: bool
    default: Any | None = None
    options: list[Any] | None = None
    description: str | None = None


class SolverSchema(BaseModel):
    id: str
    name: str
    parameters: list[ParameterDefinition]


class SolversDiscoveryResponse(BaseModel):
    solvers: list[SolverSchema]


class SolverConfigSchema(BaseModel):
    type: str = Field(..., description="Solver type discriminator (e.g., GBPI, MDN)")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Solver-specific hyperparameters"
    )


class TrainEngineRequest(BaseModel):
    dataset_name: str
    solver: SolverConfigSchema
    transforms: list[dict] = Field(default_factory=list)


class EpochMetric(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float | None = None


class TrainEngineResponse(BaseModel):
    dataset_name: str
    solver_type: str
    engine_version: int
    status: str
    duration_seconds: float
    n_train_samples: int
    n_test_samples: int
    split_ratio: float
    training_history: list[EpochMetric] = Field(default_factory=list)
    transform_summary: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    dataset_name: str
    solver_type: str = "GBPI"
    version: int | None = None
    target_objective: tuple[float, float]
    n_samples: int = Field(default=10, ge=1, le=1000)


class GenerateResponse(BaseModel):
    solver_type: str
    target_objective: tuple[float, float]
    candidate_decisions: list[list[float]]
    candidate_objectives: list[tuple[float, float]]
    best_index: int
    best_candidate_objective: tuple[float, float]
    best_candidate_decision: list[float]
    best_candidate_residual: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineListItem(BaseModel):
    dataset_name: str | None = None
    solver_type: str
    version: int
    created_at: datetime


class BulkDeleteEnginesRequest(BaseModel):
    engines: list[dict] = Field(
        ..., description="list of {dataset_name, solver_type, version}"
    )
