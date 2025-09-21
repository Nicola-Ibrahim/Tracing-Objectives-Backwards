"""Domain service orchestrating feasibility validation for objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...shared.errors import ObjectiveOutOfBoundsError
from ...shared.ndarray_utils import clip01, ensure_2d
from ...shared.reasons import FeasibilityFailureReason
from ..aggregates import FeasibilityAssessment
from ..policies.validators import (
    BaseFeasibilityValidator,
    HistoricalRangeValidator,
    ParetoProximityValidator,
    ValidationResult,
)
from ..strategies import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
    FeasibilityScoringStrategy,
)


_DIVERSITY_MAP = {
    "euclidean": ClosestPointsDiversityStrategy,
    "kmeans": KMeansDiversityStrategy,
    "max_min_distance": MaxMinDistanceDiversityStrategy,
}


@dataclass(slots=True)
class ObjectiveFeasibilityChecker:
    pareto_front: np.ndarray
    pareto_front_normalized: np.ndarray
    tolerance: float
    scorer: FeasibilityScoringStrategy

    def __post_init__(self) -> None:
        if self.pareto_front.shape != self.pareto_front_normalized.shape:
            raise ValueError("Pareto front and normalised front must share the same shape.")
        if self.pareto_front.size == 0:
            raise ValueError("Pareto front cannot be empty for feasibility checking.")

        self._min_raw = self.pareto_front.min(axis=0)
        self._max_raw = self.pareto_front.max(axis=0)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def assess(
        self,
        *,
        target: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "euclidean",
        random_seed: int | None = None,
    ) -> FeasibilityAssessment:
        target_raw = ensure_2d(target)
        target_norm = ensure_2d(target_normalized)

        validators: list[BaseFeasibilityValidator] = [
            HistoricalRangeValidator(
                target=target_raw,
                historical_min=self._min_raw,
                historical_max=self._max_raw,
            ),
            ParetoProximityValidator(
                target_normalized=target_norm,
                scorer=self.scorer,
                tolerance=self.tolerance,
                pareto_front_normalized=self.pareto_front_normalized,
            ),
        ]

        failing_result: ValidationResult | None = None
        last_result: ValidationResult | None = None

        for validator in validators:
            result = validator.validate()
            last_result = result
            if not result.is_feasible:
                failing_result = result
                break

        if failing_result is None:
            return FeasibilityAssessment(
                target_normalized=target_norm,
                is_feasible=True,
                score=last_result.score if last_result else None,
            )

        suggestions = self._generate_suggestions(
            target_normalized=target_norm,
            num_suggestions=num_suggestions,
            suggestion_noise_scale=suggestion_noise_scale,
            diversity_method=diversity_method,
            random_seed=random_seed,
        )

        return FeasibilityAssessment(
            target_normalized=target_norm,
            is_feasible=False,
            score=failing_result.score,
            reason=failing_result.reason or FeasibilityFailureReason.UNKNOWN_FEASIBILITY_ISSUE,
            suggestions=suggestions,
            diagnostics={
                "validator_extra": failing_result.extra_info or "",
            },
        )

    def validate(self, **kwargs) -> FeasibilityAssessment:
        assessment = self.assess(**kwargs)
        if not assessment.is_feasible:
            raise ObjectiveOutOfBoundsError(
                message=assessment.diagnostics.get("validator_extra", "Feasibility failed."),
                reason=assessment.reason or FeasibilityFailureReason.UNKNOWN_FEASIBILITY_ISSUE,
                score=assessment.score,
                suggestions=assessment.suggestions,
            )
        return assessment

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _generate_suggestions(
        self,
        *,
        target_normalized: np.ndarray,
        num_suggestions: int,
        suggestion_noise_scale: float,
        diversity_method: str,
        random_seed: int | None,
    ) -> np.ndarray:
        if num_suggestions <= 0 or self.pareto_front_normalized.size == 0:
            return np.empty((0, self.pareto_front_normalized.shape[1]))

        strategy_cls = _DIVERSITY_MAP.get(diversity_method, ClosestPointsDiversityStrategy)
        strategy = strategy_cls(random_seed=random_seed)
        selected = strategy.select_diverse_points(
            pareto_front_normalized=self.pareto_front_normalized,
            target_normalized=target_normalized,
            num_suggestions=num_suggestions,
        )

        if selected.size == 0:
            return selected

        rng = np.random.default_rng(random_seed)
        perturbation_range = self.tolerance * suggestion_noise_scale
        noise = rng.uniform(-perturbation_range, perturbation_range, size=selected.shape)
        return clip01(selected + noise)


__all__ = ["ObjectiveFeasibilityChecker"]
