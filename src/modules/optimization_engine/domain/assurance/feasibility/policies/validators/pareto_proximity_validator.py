"""Validator checking distance to the normalised Pareto front."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseFeasibilityValidator, ValidationResult
from ....shared.reasons import FeasibilityFailureReason
from ...strategies.scoring.base import FeasibilityScoringStrategy


@dataclass(slots=True)
class ParetoProximityValidator(BaseFeasibilityValidator):
    target_normalized: np.ndarray
    scorer: FeasibilityScoringStrategy
    tolerance: float
    pareto_front_normalized: np.ndarray

    def validate(self) -> ValidationResult:
        score = float(
            self.scorer.compute_score(
                self.target_normalized, self.pareto_front_normalized
            )
        )
        if not self.scorer.is_feasible(score, self.tolerance):
            return ValidationResult(
                is_feasible=False,
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                extra_info=f"score={score:.6f} threshold={self.tolerance:.6f}",
                score=score,
            )
        return ValidationResult(is_feasible=True, score=score)


__all__ = ["ParetoProximityValidator"]
