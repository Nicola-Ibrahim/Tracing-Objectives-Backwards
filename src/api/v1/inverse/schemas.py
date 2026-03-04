from datetime import datetime
from typing import Any, List, Tuple

from pydantic import BaseModel, Field


class SolverConfigSchema(BaseModel):
    type: str = Field(..., description="Solver type discriminator (e.g., GBPI, MDN)")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Solver-specific hyperparameters"
    )


class TrainEngineRequest(BaseModel):
    dataset_name: str
    solver: SolverConfigSchema
    transforms: List[dict] = Field(default_factory=list)
    split_ratio: float = 0.2
    random_state: int = 42


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
    loss_history: List[EpochMetric] = Field(default_factory=list)
    transform_summary: List[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    dataset_name: str
    solver_type: str = "GBPI"
    version: int | None = None
    target_objective: Tuple[float, float]
    n_samples: int = Field(default=50, ge=1, le=1000)
    trust_radius: float = Field(default=0.05, gt=0, le=1)
    concentration_factor: float = Field(default=10.0, gt=0)
    error_threshold: float | None = Field(default=None, gt=0)


class GenerateResponse(BaseModel):
    solver_type: str
    target_objective: Tuple[float, float]
    candidate_decisions: List[List[float]]
    candidate_objectives: List[Tuple[float, float]]
    best_index: int
    best_objective: Tuple[float, float]
    best_decision: List[float]
    all_residuals: List[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineListItem(BaseModel):
    solver_type: str
    version: int
    created_at: datetime
