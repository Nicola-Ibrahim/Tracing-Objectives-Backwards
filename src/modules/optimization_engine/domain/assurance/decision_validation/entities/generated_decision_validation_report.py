from pydantic import BaseModel, ConfigDict, Field

from ..enums.verdict import Verdict
from ..value_objects.gate_result import GateResult


class GeneratedDecisionValidationReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verdict: Verdict
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    explanations: dict[str, str] = Field(default_factory=dict)
    gate_results: tuple[GateResult, ...] = Field(default_factory=tuple)
