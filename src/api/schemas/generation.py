from typing import List, Literal, Tuple

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    dataset_name: str
    target_objective: Tuple[float, float]
    n_samples: int = Field(default=50, ge=1, le=1000)
    trust_radius: float = Field(default=0.05, gt=0, le=1)
    concentration_factor: float = Field(default=10.0, gt=0)


class GenerationResponse(BaseModel):
    pathway: Literal["coherent", "incoherent"]
    target_objective: Tuple[float, float]
    candidate_decisions: List[List[float]]  # M x D array of decisions
    candidate_objectives: List[Tuple[float, float]]  # M x 2 array of predictions
    residual_errors: List[float]  # M sorted errors
    anchor_indices: List[int]  # Used to plot the boundary simplex
    is_inside_mesh: bool
