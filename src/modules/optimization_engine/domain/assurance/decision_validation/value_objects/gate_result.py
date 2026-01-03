from pydantic import BaseModel, ConfigDict, Field

from ..enums.verdict import Verdict


class GateResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    passed: Verdict
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    explanation: str
