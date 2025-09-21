"""Domain layer for assurance: feasibility checks and decision validation."""

from .shared import (
    ObjectiveOutOfBoundsError,
    ValidationError,
    FeasibilityFailureReason,
)
from .feasibility import ObjectiveFeasibilityChecker, FeasibilityAssessment
from .decision_validation import (
    DecisionValidationService,
    DecisionValidationCase,
)

__all__ = [
    "ObjectiveOutOfBoundsError",
    "ValidationError",
    "FeasibilityFailureReason",
    "ObjectiveFeasibilityChecker",
    "FeasibilityAssessment",
    "DecisionValidationService",
    "DecisionValidationCase",
]
