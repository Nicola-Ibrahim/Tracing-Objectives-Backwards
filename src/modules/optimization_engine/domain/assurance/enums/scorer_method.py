from enum import Enum


class ScorerMethod(str, Enum):
    """Available scoring strategies for feasibility checks."""

    MIN_DISTANCE = "min_distance"
    KDE = "kde"
