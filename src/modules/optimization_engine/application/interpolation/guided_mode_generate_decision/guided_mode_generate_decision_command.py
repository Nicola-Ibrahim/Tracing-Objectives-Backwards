from pydantic import BaseModel, Field


class GuidedModeGenerateDecisionCommand(BaseModel):
    interpolator_name: str = Field(
        ..., description="Name of the trained interpolator to use"
    )
    target_objective: list[float] = Field(
        ..., description="User-selected point in objective space"
    )
    neighbor_count: int = Field(10, description="Number of neighbors to use")
