from dataclasses import dataclass

import numpy as np

from ..enums import FeasibilityFailureReason
from .base import BaseFeasibilityValidator, ValidationResult


@dataclass
class HistoricalObjectiveRangeValidator(BaseFeasibilityValidator):
    """
    Validates if the candidate objective point lies within the historically observed
    range of objective values from the Pareto front.
    """

    target: np.ndarray
    historical_min_values: np.ndarray
    historical_max_values: np.ndarray

    def validate(self) -> ValidationResult:
        violations = []
        for i in range(self.target.shape[1]):
            val = self.target[0, i]
            # Check against the historical min/max values
            if not (
                self.historical_min_values[i] <= val <= self.historical_max_values[i]
            ):
                violations.append(
                    f"Objective {i}: {val:.4f} not in [{self.historical_min_values[i]:.4f}, {self.historical_max_values[i]:.4f}]"
                )

        if violations:
            return ValidationResult(
                is_feasible=False,
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,  # This enum value still reflects the nature of violation
                extra_info="\n".join(violations),
                score=None,
            )
        return ValidationResult(is_feasible=True)
