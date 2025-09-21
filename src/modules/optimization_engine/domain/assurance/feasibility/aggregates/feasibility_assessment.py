"""Aggregate root describing the output of a feasibility assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ...shared.reasons import FeasibilityFailureReason


@dataclass(slots=True)
class FeasibilityAssessment:
    """Immutable view of a feasibility decision for a target objective."""

    target_normalized: np.ndarray
    is_feasible: bool
    score: float | None = None
    reason: FeasibilityFailureReason | None = None
    suggestions: np.ndarray | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - basic invariants
        if self.is_feasible and self.reason is not None:
            raise ValueError("Feasible assessments cannot include a failure reason.")
        if self.suggestions is not None and not isinstance(self.suggestions, np.ndarray):
            raise TypeError("suggestions must be an ndarray")

    @property
    def suggestion_count(self) -> int:
        if self.suggestions is None:
            return 0
        return int(self.suggestions.shape[0])


__all__ = ["FeasibilityAssessment"]
