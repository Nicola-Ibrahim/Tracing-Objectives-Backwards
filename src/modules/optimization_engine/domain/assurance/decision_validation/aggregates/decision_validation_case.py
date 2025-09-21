"""Aggregate root capturing a decision validation execution."""

from dataclasses import dataclass

from ..entities.generated_decision_validation_report import (
    GeneratedDecisionValidationReport,
    Verdict,
)
from ..value_objects.validation_outcome import ValidationOutcome


@dataclass(slots=True)
class DecisionValidationCase:
    outcome: ValidationOutcome
    report: GeneratedDecisionValidationReport

    @property
    def verdict(self) -> Verdict:
        return self.report.verdict


__all__ = ["DecisionValidationCase"]
