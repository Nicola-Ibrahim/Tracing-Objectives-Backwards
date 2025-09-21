"""Base infrastructure for feasibility validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ....shared.reasons import FeasibilityFailureReason


@dataclass(slots=True)
class ValidationResult:
    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    extra_info: str | None = None
    score: float | None = None


class BaseFeasibilityValidator(ABC):
    """Abstract base class for all feasibility validators."""

    @abstractmethod
    def validate(self) -> ValidationResult: ...


__all__ = ["ValidationResult", "BaseFeasibilityValidator"]
