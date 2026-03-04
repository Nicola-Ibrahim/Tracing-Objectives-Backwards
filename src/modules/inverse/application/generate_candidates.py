from pydantic import BaseModel, ConfigDict, Field

from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.services.generation import CandidateGenerationDomainService


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
    solver_type: str = Field(default="GBPI", description="Type of solver to use")
    version: int | None = Field(
        default=None, description="Specific engine version number to use (optional)"
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

    def model_post_init(self, __context):
        pass


class GenerateCoherentCandidatesService:
    def __init__(
        self,
        inverse_mapping_engine_repository: BaseInverseMappingEngineRepository,
        logger: BaseLogger,
    ):
        self._inverse_mapping_engine_repository = inverse_mapping_engine_repository
        self._logger = logger

    def execute(self, config: GenerationConfig) -> dict:
        self._logger.log_info(
            f"Starting generation for '{config.dataset_name}' with target {config.target_objective}"
        )

        # 1. Load context
        engine = self._inverse_mapping_engine_repository.load(
            config.dataset_name, solver_type=config.solver_type, version=config.version
        )

        # 2. Delegate the heavy lifting to the Domain Service
        result = CandidateGenerationDomainService.generate(
            target_objective=config.target_objective,
            engine=engine,
            n_samples=config.n_samples,
        )

        self._logger.log_info(
            f"Generated candidates via {result.pathway} pathway. "
            f"Winner: {result.best_objective_raw.flatten()}"
        )

        # # Return formatted dictionary
        # common_fields = {
        #     "solver_type": config.solver_type,
        #     "target_objective": result.target_objective_raw.tolist(),
        #     "candidate_decisions": result.candidate_decisions_raw.tolist(),
        #     "candidate_objectives": result.candidate_objectives_raw.tolist(),
        #     "best_index": result.best_index,
        #     "best_objective": result.best_objective_raw.flatten().tolist(),
        #     "best_decision": result.best_decision_raw.flatten().tolist(),
        #     "all_residuals": result.all_residuals.tolist(),
        # }

        # metadata = {
        #     "pathway": result.pathway,
        #     "objective_space_residual_sorted": result.objective_space_residual_sorted.tolist(),
        #     "vertices_indices": result.metadata.get("vertices_indices", []),
        #     "tau": result.metadata.get("tau"),
        #     "vertice_distances": result.metadata.get("vertice_distances", []),
        #     "is_simplex_found": result.metadata.get("is_simplex_found", False),
        #     "is_coherent": result.metadata.get("is_coherent", False),
        # }

        # # Merge results from domain service metadata if any
        # solver_metadata = result.metadata.copy()
        # # Remove already extracted fields
        # for key in [
        #     "vertices_indices",
        #     "tau",
        #     "vertice_distances",
        #     "is_simplex_found",
        #     "is_coherent",
        # ]:
        #     solver_metadata.pop(key, None)

        # metadata.update(solver_metadata)

        # common_fields["metadata"] = metadata

        return {
            "solver_type": config.solver_type,
            "target_objective": result.target_objective_raw.flatten().tolist(),
            "candidate_decisions": result.candidate_decisions_raw.tolist(),
            "candidate_objectives": [
                tuple(obj) for obj in result.candidate_objectives_raw.tolist()
            ],
            "best_index": result.best_index,
            "best_objective": result.best_objective_raw.flatten().tolist(),
            "best_decision": result.best_decision_raw.flatten().tolist(),
            "all_residuals": result.all_residuals.tolist(),
            "metadata": result.metadata,
        }
