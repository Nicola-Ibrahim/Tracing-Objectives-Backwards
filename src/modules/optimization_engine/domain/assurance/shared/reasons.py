"""Enumerations describing assurance diagnostics and failure reasons."""

from enum import Enum


class FeasibilityFailureReason(str, Enum):
    """Failure reasons emitted by feasibility validators."""

    OUT_OF_RAW_BOUNDS = "OUT_OF_RAW_BOUNDS"
    OUTSIDE_CONVEX_HULL = "OUTSIDE_CONVEX_HULL"
    TOO_FAR_FROM_FRONT = "TOO_FAR_FROM_FRONT"
    UNKNOWN_FEASIBILITY_ISSUE = "UNKNOWN_FEASIBILITY_ISSUE"


__all__ = ["FeasibilityFailureReason"]
