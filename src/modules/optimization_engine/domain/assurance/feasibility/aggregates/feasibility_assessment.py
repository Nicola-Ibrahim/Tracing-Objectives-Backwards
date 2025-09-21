"""Aggregate root describing the output of a feasibility assessment."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...shared.reasons import FeasibilityFailureReason
from ..value_objects import ObjectiveVector, Score, Suggestions


class FeasibilityAssessment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: ObjectiveVector
    is_feasible: bool
    score: Score | None = None
    reason: FeasibilityFailureReason | None = None
    suggestions: Suggestions | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reason(self):
        if self.is_feasible and self.reason is not None:
            raise ValueError("Feasible assessments cannot include a failure reason.")
        return self

    @property
    def suggestion_count(self) -> int:
        return self.suggestions.count if self.suggestions is not None else 0


__all__ = ["FeasibilityAssessment"]
