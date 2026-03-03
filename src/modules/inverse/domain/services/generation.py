from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..entities.inverse_mapping_engine import InverseMappingEngine
from .candidate_ranker import CandidateRanker


class GenerationResult(BaseModel):
    """Value object representing the complete result of a generation cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pathway: str
    target_objective_raw: np.ndarray
    candidate_decisions_raw: np.ndarray
    candidate_objectives_raw: np.ndarray
    objective_space_residual_sorted: np.ndarray
    best_index: int
    best_objective_raw: np.ndarray
    best_decision_raw: np.ndarray
    all_residuals: np.ndarray
    metadata: dict[str, Any]


class CandidateGenerationDomainService:
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

        # 4. Rank candidates
        rank_result = CandidateRanker.rank(
            candidates_X=generation_result.candidates_X,
            candidates_y=generation_result.candidates_y,
            target_objective=target_objective,
        )

        candidate_X_sorted = generation_result.candidates_X[rank_result.sort_indices]
        candidate_y_sorted = generation_result.candidates_y[rank_result.sort_indices]

        # 6. Calculate All Residuals (for frontend zoom plot)
        diffs = generation_result.candidates_y - target_objective
        all_residuals = np.linalg.norm(diffs, axis=1)

        # 7. Detransform to Raw Space
        return GenerationResult(
            pathway=generation_result.metadata.get("pathway", "unknown"),
            target_objective_raw=target_objective,
            candidate_decisions_raw=engine.detransform_decision(candidate_X_sorted),
            candidate_objectives_raw=engine.detransform_objective(candidate_y_sorted),
            best_objective_raw=engine.detransform_objective(rank_result.best_objective),
            best_decision_raw=engine.detransform_decision(rank_result.best_decision),
            objective_space_residual_sorted=rank_result.objective_space_residual_sorted,
            metadata={
                "tau": engine.solver.tau if hasattr(engine.solver, "tau") else None,
                "vertice_distances": generation_result.metadata.get(
                    "anchor_distances", []
                ),
                "is_simplex_found": generation_result.metadata.get(
                    "is_simplex_found", False
                ),
                "is_coherent": generation_result.metadata.get("is_coherent", False),
                "vertices_indices": generation_result.metadata.get(
                    "vertices_indices", []
                ),
            },
            best_index=rank_result.best_index,
            all_residuals=all_residuals,
        )
