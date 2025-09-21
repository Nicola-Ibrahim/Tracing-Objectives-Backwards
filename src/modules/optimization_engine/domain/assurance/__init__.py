"""Domain layer for assurance: feasibility checks and decision validation."""

from .shared import (
    ObjectiveOutOfBoundsError,
    ValidationError,
    FeasibilityFailureReason,
)
from .feasibility import ObjectiveFeasibilityService, FeasibilityAssessment
from .decision_validation import (
    DecisionValidationService,
    DecisionValidationCase,
)
from .interfaces import FeasibilityScoringStrategy, DiversityStrategy

__all__ = [
    "ObjectiveOutOfBoundsError",
    "ValidationError",
    "FeasibilityFailureReason",
    "ObjectiveFeasibilityService",
    "FeasibilityAssessment",
    "DecisionValidationService",
    "DecisionValidationCase",
    "FeasibilityScoringStrategy",
    "DiversityStrategy",
]
