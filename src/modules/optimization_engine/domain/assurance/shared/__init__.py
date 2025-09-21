"""Shared utilities, types, and errors for assurance domain modules."""

from .errors import ObjectiveOutOfBoundsError, ValidationError
from .reasons import FeasibilityFailureReason
from .types import ArrayLike, Score

__all__ = [
    "ObjectiveOutOfBoundsError",
    "ValidationError",
    "FeasibilityFailureReason",
    "ArrayLike",
    "Score",
]
