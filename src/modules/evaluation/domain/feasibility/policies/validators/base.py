"""Base infrastructure for feasibility validators."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from ....shared.reasons import FeasibilityFailureReason


class ValidationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    extra_info: str | None = None
    score: float | None = None


class BaseFeasibilityValidator(ABC):
    """Abstract base class for all feasibility validators."""

    @abstractmethod
    def validate(self) -> ValidationResult: ...


__all__ = ["ValidationResult", "BaseFeasibilityValidator"]
