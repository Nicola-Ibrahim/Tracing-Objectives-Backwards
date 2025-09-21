from dataclasses import dataclass

from .....enums.feasibility_failure_reason import FeasibilityFailureReason


@dataclass
class ValidationResult:
    """
    Represents the outcome of a single feasibility validation check.
    """

    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    extra_info: str | None = None
    score: float | None = None
