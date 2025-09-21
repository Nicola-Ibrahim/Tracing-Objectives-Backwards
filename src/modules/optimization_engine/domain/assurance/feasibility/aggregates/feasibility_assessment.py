"""Aggregate root describing the output of a feasibility assessment."""

from dataclasses import dataclass, field

from ...shared.reasons import FeasibilityFailureReason
from ..value_objects import ObjectiveVector, Suggestions, Score


@dataclass(slots=True)
class FeasibilityAssessment:
    """Immutable view of a feasibility decision for a target objective."""

    target: ObjectiveVector
    is_feasible: bool
    score: Score | None = None
    reason: FeasibilityFailureReason | None = None
    suggestions: Suggestions | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - basic invariants
        if self.is_feasible and self.reason is not None:
            raise ValueError("Feasible assessments cannot include a failure reason.")

    @property
    def suggestion_count(self) -> int:
        return self.suggestions.count if self.suggestions is not None else 0


__all__ = ["FeasibilityAssessment"]
