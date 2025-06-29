from pydantic import BaseModel, Field


class FreeModeGenerateDecisionCommand(BaseModel):
    interpolator_type: str = Field(..., description="Name of the interpolator to load")
    target_objective: list[float] = Field(
        ..., description="Target point in objective space (f1, f2, ...)"
    )
