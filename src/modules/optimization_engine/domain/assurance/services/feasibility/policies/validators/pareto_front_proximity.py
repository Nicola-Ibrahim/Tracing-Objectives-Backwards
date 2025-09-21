from dataclasses import dataclass

import numpy as np

from ...enums.feasibility_failure_reason import FeasibilityFailureReason
from ..scorers.base import FeasibilityScoringStrategy
from .base import BaseFeasibilityValidator, ValidationResult


@dataclass
class ParetoFrontProximityValidator(BaseFeasibilityValidator):
    """
    Validates if the candidate normalized objective point is sufficiently close to the
    historical Pareto front based on a computed score and a tolerance.
    """

    target_normalized: (
        np.ndarray
    )  # The specific normalized objective point being validated
    scorer: FeasibilityScoringStrategy
    tolerance: float
    historical_normalized_front: np.ndarray  # The normalized historical Pareto front

    def validate(self) -> ValidationResult:
        # This validator uses the normalized candidate and the historical normalized Pareto front
        score = self.scorer.compute_score(
            self.target_normalized, self.historical_normalized_front
        )

        if not self.scorer.is_feasible(score, self.tolerance):
            return ValidationResult(
                is_feasible=False,
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,  # This enum value still reflects the nature of violation
                extra_info=f"Computed score = {score:.16f}, tolerance = {self.tolerance:.4f}",
                score=score,
            )
        return ValidationResult(
            is_feasible=True, score=score
        )  # Include score even if feasible
