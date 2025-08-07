from enum import Enum


class FeasibilityFailureReason(Enum):
    """
    Enumeration of specific reasons why an objective might be deemed unfeasible.
    """

    OUT_OF_RAW_BOUNDS = "OUT_OF_RAW_BOUNDS"
    OUTSIDE_CONVEX_HULL = "OUTSIDE_CONVEX_HULL"
    TOO_FAR_FROM_FRONT = "TOO_FAR_FROM_FRONT"
    UNKNOWN_FEASIBILITY_ISSUE = "UNKNOWN_FEASIBILITY_ISSUE"  # Catch-all


class ScorerMethod(str, Enum):
    """Available scoring strategies for feasibility checks."""

    MIN_DISTANCE = "min_distance"
    KDE = "kde"
    # Add new scorer strategies here as you create them


class DiversityMethod(str, Enum):
    """Available diversity methods for generating suggestions."""

    MAX_MIN_DISTANCE = "max_min_distance"
    KMEANS = "kmeans"
    EUCLIDEAN = "euclidean"
    # Add new diversity strategies here
