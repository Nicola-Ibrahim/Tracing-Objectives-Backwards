"""Validator ensuring objectives stay within historical bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseFeasibilityValidator, ValidationResult
from ....shared.reasons import FeasibilityFailureReason
from ....shared.ndarray_utils import ensure_2d


@dataclass(slots=True)
class HistoricalRangeValidator(BaseFeasibilityValidator):
    target: np.ndarray
    historical_min: np.ndarray
    historical_max: np.ndarray

    def validate(self) -> ValidationResult:
        target = ensure_2d(self.target)
        violations: list[str] = []
        for idx, value in enumerate(target[0]):
            low = float(self.historical_min[idx])
            high = float(self.historical_max[idx])
            if not (low <= value <= high):
                violations.append(
                    f"Objective {idx}: {value:.4f} not in [{low:.4f}, {high:.4f}]"
                )

        if violations:
            return ValidationResult(
                is_feasible=False,
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                extra_info="; ".join(violations),
            )
        return ValidationResult(is_feasible=True)


__all__ = ["HistoricalRangeValidator"]
