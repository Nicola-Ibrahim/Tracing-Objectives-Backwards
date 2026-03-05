from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...dataset.domain.interfaces.base_visualizer import BaseVisualizer
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.config import GenerationConfig
from ..domain.interfaces.base_context_repository import BaseContextRepository
from ..domain.services.generation import (
    CandidateGenerationDomainService,
    GenerationResult,
)


class VisualizeGenerationUseCase:
    """
    Consolidated application usecase that handles both candidate generation
    and their visualization against the background context.
    Directly depends on the domain service and context repository.
    """

    def __init__(
        self,
        context_repository: BaseContextRepository,
        generation_domain_service: CandidateGenerationDomainService,
        dataset_repository: BaseDatasetRepository,
        visualizer: BaseVisualizer,
        logger: BaseLogger,
    ):
        self._context_repository = context_repository
        self._generation_domain_service = generation_domain_service
        self._dataset_repository = dataset_repository
        self._visualizer = visualizer
        self._logger = logger

    def execute(self, config: GenerationConfig) -> dict:
        self._logger.log_info(
            f"Executing VisualizeGenerationUseCase for dataset '{config.dataset_name}'"
        )

        # 1. Load context
        context = self._context_repository.load(config.dataset_name)

        # 2. Generate Candidates (Domain Service)
        result: GenerationResult = self._generation_domain_service.generate(
            context=context, config=config
        )

        self._logger.log_info(
            f"Generated candidates via {result.pathway} pathway. "
            f"Winner: {result.best_objective_raw.flatten()}"
        )

        # 3. Visualize Results
        self._logger.log_info(
            f"Generating context visualization plots for '{config.dataset_name}'..."
        )

        # Load original dataset to get context points for plotting
        dataset = self._dataset_repository.load(config.dataset_name)

        # Trigger visualization
        self._visualizer.plot(
            {
                "original_objectives": dataset.objectives,
                "original_decisions": dataset.decisions,
                "target_objective": config.target_objective,
                "candidate_objectives": result.candidate_objectives_raw,
                "candidate_decisions": result.candidate_decisions_raw,
                "vertices_indices": result.vertices_indices,
            }
        )

        self._logger.log_info("VisualizeGenerationUseCase completed successfully.")

        # Return formatted dictionary
        return {
            "pathway": result.pathway,
            "target_objective": result.target_objective_raw.flatten().tolist(),
            "candidate_decisions": result.candidate_decisions_raw.tolist(),
            "candidate_objectives": result.candidate_objectives_raw.tolist(),
            "y_space_residuals": result.y_space_residuals.tolist(),
            "is_simplex_found": result.is_simplex_found,
            "is_coherent": result.is_coherent,
            "best_index": result.best_index,
            "best_objective": result.best_objective_raw.flatten().tolist(),
            "best_decision": result.best_decision_raw.flatten().tolist(),
            "vertices_indices": result.vertices_indices,
        }
