import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...dataset.domain.interfaces.base_visualizer import BaseVisualizer
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_context_repository import BaseContextRepository
from ..domain.services.barycentric_locator import BarycentricLocator
from ..domain.services.candidate_ranker import CandidateRanker, RankingResult
from ..domain.services.dirichlet_sampler import DirichletSampler
from ..infrastructure.optimizers.trust_region import TrustRegionOptimizer


class GenerationConfig(BaseModel):
    """
    User-configurable parameters for the generation pipeline.
    Combines dynamic execution parameters and coherence tolerances.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Name of the dataset")
    target_objective: tuple[float, float] = Field(
        ..., description="Target objective coordinates (2D)"
    )
    n_samples: int = Field(
        default=50, ge=1, description="Number of Dirichlet weight samples (Phase 3A)"
    )
    concentration_factor: float = Field(
        default=10.0,
        gt=0,
        description="Controls tightness of Dirichlet sampling around target",
    )
    trust_radius: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="Trust-region radius in normalized decision space (Phase 3B)",
    )
    error_threshold: float | None = Field(
        default=None,
        description="Residual error cutoff for filtering; None = no filtering",
    )

    def model_post_init(self, __context):
        if self.error_threshold is not None and self.error_threshold <= 0:
            raise ValueError(
                "error_threshold must be strictly greater than 0 if provided"
            )


class GenerateCoherentCandidatesService:
    """
    Orchestrates the generation of physically coherent candidates for a requested target objective.
    Implements both Phase 3A (Coherent) and Phase 3B (Incoherent) pathways.
    Now handles normalization automatically using the context's localized normalizers.
    """

    def __init__(
        self,
        context_repository: BaseContextRepository,
        dataset_repository: BaseDatasetRepository,
        logger: BaseLogger,
        visualizer: BaseVisualizer | None = None,
    ):
        self._context_repository = context_repository
        self._dataset_repository = dataset_repository
        self._logger = logger
        self._visualizer = visualizer

    def execute(self, config: GenerationConfig) -> RankingResult:
        self._logger.log_info(
            f"Starting generation for '{config.dataset_name}' with target {config.target_objective}"
        )
        target_arr_raw = np.array(config.target_objective).reshape(1, -1)

        # 1. Load context
        context = self._context_repository.load(config.dataset_name)

        # 2. Apply transform steps to target objective for the optimization steps
        target_arr_norm = target_arr_raw.copy()
        for t in context.get_objectives_transforms():
            target_arr_norm = t.transform(target_arr_norm)

        # Normalize the stored raw context objectives to find the closest anchor properly
        context_objectives_norm = context.objectives.copy()
        for t in context.get_objectives_transforms():
            context_objectives_norm = t.transform(context_objectives_norm)

        # 3. Localize using normalized objectives
        anchor_indices, weights, is_inside = BarycentricLocator.locate(
            target_arr_norm, context_objectives_norm
        )

        if not is_inside:
            self._logger.log_warning(
                "Target is outside the Delaunay mesh. Using nearest fallback."
            )

        anchors_norm = context.anchors_norm[anchor_indices]

        # 4. Check Reliability
        is_coherent = context.evaluate_simplex_reliability(anchors_norm)

        if is_coherent:
            self._logger.log_info(
                "Region is coherent. Proceeding with Dirichlet sampling."
            )
            pathway = "coherent"
            weight_samples = DirichletSampler.sample(
                weights, config.n_samples, config.concentration_factor
            )
            candidate_decisions_norm = np.dot(weight_samples, anchors_norm)
        else:
            self._logger.log_warning(
                "Incoherent region detected. Falling back to Trust-Region optimization."
            )
            pathway = "incoherent"

            # Phase 3B: Find closest anchor in objective space
            anchor_objs_norm = context_objectives_norm[anchor_indices]
            dists = np.linalg.norm(anchor_objs_norm - target_arr_norm, axis=1)
            closest_local_idx = int(np.argmin(dists))
            base_anchor = anchors_norm[closest_local_idx]

            candidate_decisions_norm = TrustRegionOptimizer.optimize(
                surrogate=context.surrogate_step,
                base_anchor=base_anchor,
                target_objective=target_arr_norm,
                trust_radius=config.trust_radius,
                n_candidates=config.n_samples,
            )

        # 5. Evaluate via Surrogate (all in norm space)
        predicted_objectives_norm = context.surrogate_step.predict(
            candidate_decisions_norm
        )

        # 6. Filter & Rank (Result is UN-NORMALIZED)
        result = CandidateRanker.rank(
            candidates=candidate_decisions_norm,
            predicted_objectives=predicted_objectives_norm,
            target_objective=target_arr_norm,
            pathway=pathway,
            anchor_indices=anchor_indices,
            is_inside_mesh=is_inside,
            objective_transforms=context.get_objectives_transforms(),
            decision_transforms=context.get_decisions_transforms(),
            error_threshold=config.error_threshold,
        )

        self._logger.log_info(
            f"Generated {len(result.candidates)} un-normalized candidates. Winner: {result.winner_point.flatten()}"
        )

        # 7. Plot results if visualizer is available
        if self._visualizer:
            self._logger.log_info("Generating context visualization plots...")

            # Use DatasetRepository to get original raw data
            dataset = self._dataset_repository.load(config.dataset_name)

            self._visualizer.plot(
                {
                    "original_objectives": dataset.objectives,
                    "original_decisions": dataset.decisions,
                    "target_objective": config.target_objective,
                    "candidate_objectives": result.predicted_objectives,
                    "candidate_decisions": result.candidates,
                    "anchor_indices": result.anchor_indices,
                }
            )

        return {
            "pathway": pathway,
            "target_objective": tuple(target_arr_raw.flatten().tolist()),
            "candidate_decisions": [row.tolist() for row in result.candidates],
            "candidate_objectives": [
                tuple(row) for row in result.predicted_objectives.tolist()
            ],
            "residual_errors": result.residual_errors.tolist(),
            "anchor_indices": result.anchor_indices,
            "is_inside_mesh": result.is_inside_mesh,
            "winner_index": result.winner_index,
            "winner_point": tuple(result.winner_point.flatten().tolist()),
            "winner_decision": result.winner_decision.flatten().tolist(),
        }
