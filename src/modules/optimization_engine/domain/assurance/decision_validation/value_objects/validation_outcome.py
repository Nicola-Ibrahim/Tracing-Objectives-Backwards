from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .gate_result import GateResult


class Verdict(StrEnum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"


class ValidationOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verdict: Verdict
    gate_results: tuple[GateResult, ...] = Field(default_factory=tuple)
