import numpy as np

from ....modeling.domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.interfaces.base_context_repository import BaseContextRepository
from ...domain.services.barycentric_locator import BarycentricLocator
from ...domain.services.candidate_ranker import CandidateRanker
from ...domain.services.coherence_gate import CoherenceGate
from ...domain.services.dirichlet_sampler import DirichletSampler
from ...domain.value_objects.coherence_params import CoherenceParams
from ...domain.value_objects.generation_result import GenerationResult
from ...infrastructure.optimizers.trust_region import TrustRegionOptimizer


class GenerateCoherentCandidatesService:
    """
    Orchestrates the generation of physically coherent candidates for a requested target objective.
    Implements both Phase 3A (Coherent) and Phase 3B (Incoherent) pathways.
    Now handles normalization automatically via TrainedPipeline.
    """

    def __init__(
        self,
        context_repository: BaseContextRepository,
        model_repository: BaseTrainedPipelineRepository,
        logger: BaseLogger,
    ):
        self._context_repository = context_repository
        self._model_repository = model_repository
        self._logger = logger

    def execute(
        self,
        dataset_name: str,
        target_objective: tuple[float, float],
        params: CoherenceParams,
    ) -> GenerationResult:
        self._logger.log_info(
            f"Starting generation for '{dataset_name}' with target {target_objective}"
        )
        target_arr_raw = np.array(target_objective).reshape(1, -1)

        # 1. Load context and surrogate (pipeline)
        context = self._context_repository.load(dataset_name)

        pipeline = self._model_repository.get_version_by_number(
            estimator_type=context.surrogate_type,
            version=context.surrogate_version,
            mapping_direction="forward",
            dataset_name=dataset_name,
        )

        # 2. Normalize target objective for the optimization steps
        target_arr_norm = target_arr_raw.copy()
        for t in pipeline.get_objectives_transforms():
            target_arr_norm = t.transform(target_arr_norm)

        # Normalize the stored raw context objectives to find the closest anchor properly
        context_objectives_norm = context.objectives.copy()
        for t in pipeline.get_objectives_transforms():
            context_objectives_norm = t.transform(context_objectives_norm)

        # 3. Localize using normalized objectives
        anchor_indices, weights, is_inside = BarycentricLocator.locate(
            target_arr_norm, context_objectives_norm
        )

        if not is_inside:
            self._logger.log_warn(
                "Target is outside the Delaunay mesh. Using nearest fallback."
            )

        anchors_norm = context.anchors_norm[anchor_indices]

        # 4. Check Coherence
        is_coherent = CoherenceGate.check(anchors_norm, context.tau)

        if is_coherent:
            self._logger.log_info(
                "Region is coherent. Proceeding with Dirichlet sampling."
            )
            pathway = "coherent"
            weight_samples = DirichletSampler.sample(
                weights, params.n_samples, params.concentration_factor
            )
            candidate_decisions_norm = np.dot(weight_samples, anchors_norm)
        else:
            self._logger.log_warn(
                "Incoherent region detected. Falling back to Trust-Region optimization."
            )
            pathway = "incoherent"

            # Phase 3B: Find closest anchor in objective space
            anchor_objs_norm = context_objectives_norm[anchor_indices]
            dists = np.linalg.norm(anchor_objs_norm - target_arr_norm, axis=1)
            closest_local_idx = int(np.argmin(dists))
            base_anchor = anchors_norm[closest_local_idx]

            candidate_decisions_norm = TrustRegionOptimizer.optimize(
                surrogate=pipeline.model.fitted,
                base_anchor=base_anchor,
                target_objective=target_arr_norm,
                trust_radius=params.trust_radius,
                n_candidates=params.n_samples,
            )

        # 5. Evaluate via Surrogate (all in norm space)
        predicted_objectives_norm = pipeline.model.fitted.predict(
            candidate_decisions_norm
        )

        # 6. Filter & Rank (in norm space)
        result = CandidateRanker.rank(
            candidates=candidate_decisions_norm,
            predicted_objectives=predicted_objectives_norm,
            target_objective=target_arr_norm,
            pathway=pathway,
            anchor_indices=anchor_indices,
            is_inside_mesh=is_inside,
            error_threshold=params.error_threshold,
        )

        # 7. Un-normalize the accepted candidates and their predicted objectives
        accepted_candidates_raw = result.candidates.copy()
        accepted_predictions_raw = result.predicted_objectives.copy()

        for t in reversed(pipeline.get_decisions_transforms()):
            accepted_candidates_raw = t.inverse_transform(accepted_candidates_raw)

        for t in reversed(pipeline.get_objectives_transforms()):
            accepted_predictions_raw = t.inverse_transform(accepted_predictions_raw)

        result.candidates = accepted_candidates_raw
        result.predicted_objectives = accepted_predictions_raw

        self._logger.log_info(
            f"Generated {len(result.candidates)} un-normalized candidates after ranking."
        )
        return result
