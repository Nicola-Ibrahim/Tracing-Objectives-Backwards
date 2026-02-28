import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RankingResult(BaseModel):
    """
    The ranked output of the generation pipeline.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    candidates_decisions_sorted: np.ndarray = Field(
        ..., description="(M, D) Final decision-space candidates (Raw/Un-normalized)"
    )
    candidates_objectives_sorted: np.ndarray = Field(
        ..., description="(M, 2) Predicted objectives (Raw/Un-normalized)"
    )
    objective_space_residual_sorted: np.ndarray = Field(
        ...,
        description="(M,) Array of residual errors vs target (Calculated in Normalized space)",
    )
    best_index: int = Field(
        ..., description="Index of the best candidate in this result"
    )
    best_objective: np.ndarray = Field(
        ..., description="(1, 2) The best performance vector (Raw/Un-normalized)"
    )
    best_decision: np.ndarray = Field(
        ..., description="(1, D) The best decision vector (Raw/Un-normalized)"
    )

    def __post_init__(self):
        num_candidates = self.candidates_decisions_sorted.shape[0]
        if self.candidates_objectives_sorted.shape[0] != num_candidates:
            raise ValueError(
                "candidates and predicted_objectives must have same number of rows"
            )
        if self.objective_space_residual_sorted.shape[0] != num_candidates:
            raise ValueError(
                "candidates and residual_errors must have same number of rows"
            )

        # Verify residual_errors are sorted ascending
        if not np.all(
            self.objective_space_residual_sorted[:-1]
            <= self.objective_space_residual_sorted[1:]
        ):
            raise ValueError(
                "RankingResult candidates must be sorted by residual error ascending"
            )


class CandidateRanker:
    """
    Domain service for comparing predicted objectives against a target and ranking candidates.
    """

    @staticmethod
    def rank(
        candidates_decisions: np.ndarray,
        candidates_objectives: np.ndarray,
        target_objective: np.ndarray,
        residual_threshold: float | None = None,
    ) -> RankingResult:
        """
        Ranks candidates by residual error, filters if needed, and un-normalizes results.

        Args:
            candidates_decisions: (M, D) array of generated candidates decisions.
            candidates_objectives: (M, 2) array of surrogate-predicted objectives.
            target_objective: (1, 2) or (2,) array of the requested target.
            residual_threshold: Optional cutoff for maximum residual error.

        Returns:
            RankingResult containing ranked, UN-NORMALIZED candidates.
        """
        target = np.asarray(target_objective).reshape(1, candidates_objectives.shape[1])

        # 1. Calculate Euclidean residual errors between predicted and target objectives
        objective_space_residual = np.linalg.norm(
            candidates_objectives - target, axis=1
        )

        # 2. Filter by error threshold
        if residual_threshold is not None:
            valid_mask = objective_space_residual <= residual_threshold
            candidates_decisions = candidates_decisions[valid_mask]
            candidates_objectives = candidates_objectives[valid_mask]
            objective_space_residual = objective_space_residual[valid_mask]

        # 3. Sort by residual error
        sort_indices = np.argsort(objective_space_residual)
        candidates_decisions_sorted = candidates_decisions[sort_indices]
        candidates_objectives_sorted = candidates_objectives[sort_indices]
        objective_space_residual_sorted = objective_space_residual[sort_indices]

        # 4. Identify Winner
        winner_idx = 0  # Ranking is already sorted ascending by error
        best_objective = candidates_objectives_sorted[winner_idx].reshape(1, 2)
        best_decision = candidates_decisions_sorted[winner_idx].reshape(1, -1)
        original_winner_index = int(sort_indices[0])

        return RankingResult(
            candidates_decisions_sorted=candidates_decisions_sorted,
            candidates_objectives_sorted=candidates_objectives_sorted,
            objective_space_residual_sorted=objective_space_residual_sorted,
            best_index=original_winner_index,
            best_objective=best_objective,
            best_decision=best_decision,
        )
