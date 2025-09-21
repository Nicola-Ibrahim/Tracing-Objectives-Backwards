"""Optional entity capturing findings from individual feasibility validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...shared.reasons import FeasibilityFailureReason


@dataclass(slots=True)
class AssessmentFinding:
    """Outcome of a single feasibility rule evaluation."""

    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    score: float | None = None
    message: str | None = None


__all__ = ["AssessmentFinding"]
