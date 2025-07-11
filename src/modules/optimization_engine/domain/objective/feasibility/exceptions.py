import numpy as np

from .enums import FeasibilityFailureReason


class ObjectiveOutOfBoundsError(Exception):
    """
    Custom exception raised when a target objective is deemed out of bounds
    or not feasible by the ObjectiveFeasibilityChecker.
    """

    def __init__(
        self,
        message: str,
        reason: FeasibilityFailureReason,
        score: float | None = None,
        suggestions: np.ndarray | None = None,
        extra_info: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.score = score
        self.suggestions = suggestions
        self.extra_info = extra_info

    def __str__(self):
        # Override str for clearer default printing if not specifically handled
        details = [f"Reason: {self.reason.value}", self.message]
        if self.score is not None:
            details.append(f"Score value: {self.score:.4f}")
        if self.extra_info:
            details.append(self.extra_info)
        return "\n".join(details)
