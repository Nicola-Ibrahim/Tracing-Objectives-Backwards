"""Domain-level error types used across assurance components."""

from dataclasses import dataclass

import numpy as np

from ..shared.errors import AssuranceError
from ..shared.reasons import FeasibilityFailureReason


@dataclass(slots=True)
class ObjectiveOutOfBoundsError(AssuranceError):
    """Raised when a target objective violates feasibility constraints."""

    message: str
    reason: FeasibilityFailureReason
    score: float | None = None
    suggestions: np.ndarray | None = None
    extra_info: str | None = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.suggestions is not None and not isinstance(
            self.suggestions, np.ndarray
        ):
            raise TypeError("suggestions must be an ndarray or None")

    def __str__(self) -> str:
        segments: list[str] = [f"Reason: {self.reason.value}", self.message]
        if self.score is not None:
            segments.append(f"Score: {self.score:.4f}")
        if self.extra_info:
            segments.append(self.extra_info)
        return " | ".join(segments)
