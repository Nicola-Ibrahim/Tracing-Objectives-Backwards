import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RankingResult(BaseModel):
    """
    The ranked output of the generation pipeline.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sort_indices: np.ndarray = Field(
        ...,
        description="(M,) Array of indices that sort the candidates by residual error",
    )
    best_index: int = Field(
        ..., description="Index of the best candidate in this result"
    )
    best_candidate_residual: float = Field(
        ..., description="Residual error of the best candidate"
    )

    def __post_init__(self):
        pass


class CandidatesRanker:
    """
    Domain service for comparing predicted objectives against a target
    and ranking candidates.
    """

    @staticmethod
    def rank(candidates_y: np.ndarray, target_objective: np.ndarray) -> RankingResult:
        """
        Ranks candidates by residual error.

        Args:
            candidates_y: (M, 2) array of surrogate-predicted objectives.
            target_objective: (1, 2) or (2,) array of the requested target.

        Returns:
            RankingResult containing ranked candidates.
        """
        target = np.asarray(target_objective).reshape(1, candidates_y.shape[1])

        # 1. Calculate Euclidean residual errors between predicted and target objectives
        y_space_residuals = np.linalg.norm(candidates_y - target, axis=1)

        # 2. Sort EVERYTHING by residual error first (identify global best)
        sort_indices = np.argsort(y_space_residuals)

        # Identify the global winner (best index in original unsorted arrays)
        best_index = int(sort_indices[0])

        return RankingResult(
            sort_indices=sort_indices,
            best_index=best_index,
            best_candidate_residual=y_space_residuals[best_index],
        )
