import numpy as np
from pydantic import BaseModel, ConfigDict

from ...infrastructure.sampling.dirichlet import DirichletSampling
from ...infrastructure.sampling.gd import GradientDescentSampling
from ..config import GenerationConfig
from ..entities.generation_context import GenerationContext
from .candidate_ranker import CandidateRanker


class GenerationResult(BaseModel):
    """Value object representing the complete result of a generation cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pathway: str
    target_objective_raw: np.ndarray
    candidate_decisions_raw: np.ndarray
    candidate_objectives_raw: np.ndarray
    objective_space_residual_sorted: np.ndarray
    is_simplex_found: bool
    is_coherent: bool
    best_index: int
    best_objective_raw: np.ndarray
    best_decision_raw: np.ndarray
    vertices_indices: list[int]


class CandidateGenerationDomainService:
    """
    Pure domain service responsible for the algorithmic generation,
    prediction, ranking, and detransformation of candidates.
    """

    def generate(
        self, context: GenerationContext, config: GenerationConfig
    ) -> GenerationResult:
        target_arr_raw = np.array(config.target_objective).reshape(1, -1)

        # 1. Normalize Target
        target_arr_norm = context.normalize_target(target_arr_raw)

        # 2. Locate in Mesh
        vertices_indices, weights, is_simplex_found, is_coherent = context.locate(
            target_arr_norm
        )

        # 3. Sample
        if is_coherent and is_simplex_found:
            pathway = "coherent"
            candidate_decisions_norm = DirichletSampling(
                concentration_factor=config.concentration_factor
            ).sample(
                vertices=context.decision_vertices[vertices_indices],
                weights=weights,
                n_samples=config.n_samples,
            )
        else:
            pathway = "incoherent"
            candidate_decisions_norm = GradientDescentSampling(
                surrogate_estimator=context.surrogate_estimator,
                target_norm=target_arr_norm,
                trust_radius=config.trust_radius,
            ).sample(
                vertices=context.decision_vertices[vertices_indices],
                weights=weights,
                n_samples=config.n_samples,
            )

        # 4. Predict
        candidates_objectives_norm = context.surrogate_estimator.predict(
            candidate_decisions_norm
        )

        # 5. Rank
        rank_result = CandidateRanker.rank(
            candidates_decisions=candidate_decisions_norm,
            candidates_objectives=candidates_objectives_norm,
            target_objective=target_arr_norm,
            residual_threshold=config.error_threshold,
        )

        # 6. Detransform to Raw Space
        return GenerationResult(
            pathway=pathway,
            target_objective_raw=target_arr_raw,
            candidate_decisions_raw=context.decision_detransform(
                rank_result.candidates_decisions_sorted
            ),
            candidate_objectives_raw=context.objective_detransform(
                rank_result.candidates_objectives_sorted
            ),
            objective_space_residual_sorted=rank_result.objective_space_residual_sorted,
            is_simplex_found=is_simplex_found,
            is_coherent=is_coherent,
            best_index=rank_result.best_index,
            best_objective_raw=context.objective_detransform(
                rank_result.best_objective
            ),
            best_decision_raw=context.decision_detransform(rank_result.best_decision),
            vertices_indices=vertices_indices,
        )
