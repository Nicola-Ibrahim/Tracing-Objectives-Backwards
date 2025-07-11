from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..enums import FeasibilityFailureReason


@dataclass
class ValidationResult:
    """
    Represents the outcome of a single feasibility validation check.
    """

    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    extra_info: str | None = None
    score: float | None = None


class BaseFeasibilityValidator(ABC):
    """
    Abstract Base Class for all feasibility validators.
    Each concrete validator will perform a specific check and return a ValidationResult.
    """

    @abstractmethod
    def validate(
        self,
    ) -> ValidationResult:
        """s
        Performs the specific feasibility validation check.

        Returns:
            ValidationResult: The outcome of the validation.
        """
        pass
