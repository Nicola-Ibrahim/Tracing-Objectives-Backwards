"""Gate result value object."""

from pydantic import BaseModel, ConfigDict, Field


class GateResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    passed: bool
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    explanation: str
