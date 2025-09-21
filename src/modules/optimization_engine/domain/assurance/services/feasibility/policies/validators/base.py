from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...enums.feasibility_failure_reason import FeasibilityFailureReason




class BaseFeasibilityValidator(ABC):
    """
    Abstract Base Class for all feasibility validators.
    Each concrete validator will perform a specific check and return a ValidationResult.
    """

    @abstractmethod
    def validate(self) -> ValidationResult:
        """s
        Performs the specific feasibility validation check.

        Returns:
            ValidationResult: The outcome of the validation.
        """
        pass
