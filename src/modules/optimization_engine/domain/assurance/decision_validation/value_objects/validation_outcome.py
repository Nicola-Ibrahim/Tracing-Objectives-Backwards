from pydantic import BaseModel, ConfigDict, Field

from ..entities.generated_decision_validation_report import Verdict
from .gate_result import GateResult


class ValidationOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verdict: Verdict
    gate_results: tuple[GateResult, ...] = Field(default_factory=tuple)


__all__ = ["ValidationOutcome"]
