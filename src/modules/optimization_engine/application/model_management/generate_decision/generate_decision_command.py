from pydantic import BaseModel, Field


class GenerateDecisionCommand(BaseModel):
    model_type: str = Field(..., description="Name of the interpolator to load")
    target_objective: list[float] = Field(
        ..., description="Target point in objective space (f1, f2, ...)"
    )
    distance_tolerance: float = 0.02
    num_suggestions: int = 3
