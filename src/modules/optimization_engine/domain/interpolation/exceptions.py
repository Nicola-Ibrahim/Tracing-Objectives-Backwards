from enum import Enum

import numpy as np


class FeasibilityFailureReason(Enum):
    """
    Enumeration of specific reasons why an objective might be deemed unfeasible.
    """

    OUT_OF_RAW_BOUNDS = "OUT_OF_RAW_BOUNDS"
    OUTSIDE_CONVEX_HULL = "OUTSIDE_CONVEX_HULL"
    TOO_FAR_FROM_FRONT = "TOO_FAR_FROM_FRONT"
    UNKNOWN_FEASIBILITY_ISSUE = "UNKNOWN_FEASIBILITY_ISSUE"  # Catch-all


class ObjectiveOutOfBoundsError(Exception):
    """
    Custom exception raised when a target objective is deemed out of bounds
    or not feasible by the ObjectiveFeasibilityChecker.
    """

    def __init__(
        self,
        message: str,
        reason: FeasibilityFailureReason,
        distance: float | None = None,
        suggestions: np.ndarray | None = None,
        extra_info: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.distance = distance
        self.suggestions = suggestions
        self.extra_info = extra_info

    def __str__(self):
        # Override str for clearer default printing if not specifically handled
        details = [f"Reason: {self.reason.value}", self.message]
        if self.distance is not None:
            details.append(f"Distance: {self.distance:.4f}")
        if self.extra_info:
            details.append(self.extra_info)
        return "\n".join(details)
