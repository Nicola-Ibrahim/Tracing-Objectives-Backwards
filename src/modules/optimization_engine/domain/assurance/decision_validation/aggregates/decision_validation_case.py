"""Aggregate root capturing a decision validation execution."""

from pydantic import BaseModel, ConfigDict

from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
)
from ..value_objects.validation_outcome import ValidationOutcome, Verdict


class DecisionValidationCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    outcome: ValidationOutcome
    report: GeneratedDecisionValidationReport

    @property
    def verdict(self) -> Verdict:
        return self.report.verdict


__all__ = ["DecisionValidationCase"]
