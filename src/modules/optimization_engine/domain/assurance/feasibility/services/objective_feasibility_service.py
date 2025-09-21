"""Domain service orchestrating feasibility validation for objectives."""

from typing import Dict, Type

import numpy as np

from ...shared.errors import ObjectiveOutOfBoundsError
from ...shared.ndarray_utils import clip01
from ...shared.reasons import FeasibilityFailureReason
from ..aggregates import FeasibilityAssessment
from ..interfaces import DiversityStrategy, FeasibilityScoringStrategy
from ..policies.validators import (
    BaseFeasibilityValidator,
    HistoricalRangeValidator,
    ParetoProximityValidator,
    ValidationResult,
)
from ..value_objects import ObjectiveVector, ParetoFront, Score, Suggestions


class ObjectiveFeasibilityService:
    """Application of feasibility business rules around provided value objects."""

    def __init__(
        self,
        *,
        scorer: FeasibilityScoringStrategy,
        diversity_registry: Dict[str, Type[DiversityStrategy]],
    ) -> None:
        if not diversity_registry:
            raise ValueError("diversity_registry must include at least one strategy.")
        self._scorer = scorer
        self._diversity_registry = dict(diversity_registry)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def assess(
        self,
        *,
        pareto_front: ParetoFront,
        tolerance: float,
        target: ObjectiveVector,
        num_suggestions: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "euclidean",
        random_seed: int | None = None,
    ) -> FeasibilityAssessment:
        min_raw, max_raw = pareto_front.bounds()

        validators: list[BaseFeasibilityValidator] = [
            HistoricalRangeValidator(
                target=target.raw,
                historical_min=min_raw,
                historical_max=max_raw,
            ),
            ParetoProximityValidator(
                target_normalized=target.normalized,
                scorer=self._scorer,
                tolerance=tolerance,
                pareto_front_normalized=pareto_front.normalized,
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
            score_vo = (
                Score(last_result.score)
                if last_result and last_result.score is not None
                else None
            )
            return FeasibilityAssessment(
                target=target,
                is_feasible=True,
                score=score_vo,
            )

        suggestions_array = self._generate_suggestions(
            pareto_front=pareto_front,
            target_normalized=target.normalized,
            num_suggestions=num_suggestions,
            suggestion_noise_scale=suggestion_noise_scale,
            diversity_method=diversity_method,
            random_seed=random_seed,
            tolerance=tolerance,
        )
        suggestions_vo = (
            Suggestions(suggestions_array)
            if suggestions_array is not None and suggestions_array.size
            else None
        )
        score_vo = (
            Score(failing_result.score) if failing_result.score is not None else None
        )

        return FeasibilityAssessment(
            target=target,
            is_feasible=False,
            score=score_vo,
            reason=failing_result.reason
            or FeasibilityFailureReason.UNKNOWN_FEASIBILITY_ISSUE,
            suggestions=suggestions_vo,
            diagnostics={
                "validator_extra": failing_result.extra_info or "",
            },
        )

    def validate(
        self,
        *,
        pareto_front: ParetoFront,
        tolerance: float,
        target: ObjectiveVector,
        num_suggestions: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "euclidean",
        random_seed: int | None = None,
    ) -> FeasibilityAssessment:
        assessment = self.assess(
            pareto_front=pareto_front,
            tolerance=tolerance,
            target=target,
            num_suggestions=num_suggestions,
            suggestion_noise_scale=suggestion_noise_scale,
            diversity_method=diversity_method,
            random_seed=random_seed,
        )
        if not assessment.is_feasible:
            raise ObjectiveOutOfBoundsError(
                message=assessment.diagnostics.get(
                    "validator_extra", "Feasibility failed."
                ),
                reason=assessment.reason
                or FeasibilityFailureReason.UNKNOWN_FEASIBILITY_ISSUE,
                score=assessment.score.value if assessment.score else None,
                suggestions=(
                    assessment.suggestions.values if assessment.suggestions else None
                ),
            )
        return assessment

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _generate_suggestions(
        self,
        *,
        pareto_front: ParetoFront,
        target_normalized: np.ndarray,
        num_suggestions: int,
        suggestion_noise_scale: float,
        diversity_method: str,
        random_seed: int | None,
        tolerance: float,
    ) -> np.ndarray | None:
        if num_suggestions <= 0 or pareto_front.normalized.size == 0:
            return None

        strategy_cls = self._diversity_registry.get(diversity_method)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown diversity method '{diversity_method}'. Registered: {list(self._diversity_registry)}"
            )
        strategy = strategy_cls(random_seed=random_seed)
        selected = strategy.select_diverse_points(
            pareto_front_normalized=pareto_front.normalized,
            target_normalized=target_normalized,
            num_suggestions=num_suggestions,
        )

        if selected.size == 0:
            return None

        rng = np.random.default_rng(random_seed)
        perturbation_range = tolerance * suggestion_noise_scale
        noise = rng.uniform(
            -perturbation_range, perturbation_range, size=selected.shape
        )
        return clip01(selected + noise)


__all__ = ["ObjectiveFeasibilityService"]
