from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..value_objects.gate_result import GateResult


class GeneratedDecisionValidationReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: dict[str, Any] = Field(default_factory=dict)
    explanations: dict[str, str] = Field(default_factory=dict)
    gate_results: tuple[GateResult, ...] = ()
