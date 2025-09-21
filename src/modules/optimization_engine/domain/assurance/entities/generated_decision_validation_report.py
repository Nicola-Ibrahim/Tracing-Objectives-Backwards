from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"


@dataclass
class GeneratedDecisionValidationReport:
    """
    Structured result from the validation pipeline.

    - verdict: "ACCEPT" if both gates pass; "ABSTAIN" otherwise.
    - metrics: numeric diagnostics (normalised units unless noted).
    - explanations: short, human-readable reasons per gate.
    """

    verdict: Verdict
    metrics: dict[str, Any] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)
