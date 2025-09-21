from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"


@dataclass(slots=True)
class GeneratedDecisionValidationReport:
    verdict: Verdict
    metrics: dict[str, Any] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)


__all__ = ["GeneratedDecisionValidationReport", "Verdict"]
