from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..entities.inverse_mapping_engine import InverseMappingEngine
from .ranking import CandidatesRanker


class GenerationResult(BaseModel):
    """Value object representing the complete result of a generation cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate_decisions: np.ndarray
    candidate_objectives: np.ndarray
    best_index: int
    best_candidate_objective: np.ndarray
    best_candidate_decision: np.ndarray
    best_candidate_residual: float
    metadata: dict[str, Any]


class CandidateGeneration:
    """
    Pure domain service responsible for the algorithmic generation,
    prediction, ranking, and detransformation of candidates.
    """

    @staticmethod
    def generate(
        target_objective: tuple[float, float],
        engine: InverseMappingEngine,
        n_samples: int,
    ) -> GenerationResult:
        target_objective = np.array(target_objective).reshape(1, -1)

        # 1. Normalize the incoming target using engine's transforms
        target_objective_norm = engine.transform_objective(target_objective)

        # 2. Generate candidates using the engine's solver
        generation_result = engine.solver.generate(
            target_y=target_objective_norm, n_samples=n_samples
        )

        # 3. Prepare Metadata
        metadata = generation_result.metadata.copy()
        
        # 4. Rank candidates
        rank_result = CandidatesRanker.rank(
            candidates_y=generation_result.candidates_y,
            target_objective=target_objective_norm,
        )

        # 5. Detransform to Raw Space
        return GenerationResult(
            candidate_decisions=engine.inverse_transform_decision(
                generation_result.candidates_X
            ),
            candidate_objectives=engine.inverse_transform_objective(
                generation_result.candidates_y
            ),
            best_index=rank_result.best_index,
            best_candidate_objective=engine.inverse_transform_objective(
                generation_result.candidates_y[rank_result.best_index].reshape(1, -1)
            ),
            best_candidate_decision=engine.inverse_transform_decision(
                generation_result.candidates_X[rank_result.best_index].reshape(1, -1)
            ),
            best_candidate_residual=rank_result.best_candidate_residual,
            metadata=metadata,
        )
