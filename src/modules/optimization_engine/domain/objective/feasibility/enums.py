from enum import Enum


class FeasibilityFailureReason(Enum):
    """
    Enumeration of specific reasons why an objective might be deemed unfeasible.
    """

    OUT_OF_RAW_BOUNDS = "OUT_OF_RAW_BOUNDS"
    OUTSIDE_CONVEX_HULL = "OUTSIDE_CONVEX_HULL"
    TOO_FAR_FROM_FRONT = "TOO_FAR_FROM_FRONT"
    UNKNOWN_FEASIBILITY_ISSUE = "UNKNOWN_FEASIBILITY_ISSUE"  # Catch-all
