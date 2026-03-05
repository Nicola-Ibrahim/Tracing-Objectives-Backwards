from pydantic import BaseModel, ConfigDict, Field

from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.services.generation import CandidateGeneration


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


class GenerateCandidatesService:
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

        # 2. Generate candidates using the engine's solver
        result = CandidateGeneration.generate(
            engine=engine,
            target_objective=config.target_objective,
            n_samples=config.n_samples,
        )

        self._logger.log_info(
            f"Generated {result.candidate_objectives.shape[0]} candidates. "
            f"Winner @{result.best_index}: {result.best_candidate_objective.flatten()}"
        )

        return {
            "solver_type": config.solver_type,
            "target_objective": config.target_objective,
            "candidate_decisions": result.candidate_decisions.tolist(),
            "candidate_objectives": [
                tuple(obj) for obj in result.candidate_objectives.tolist()
            ],
            "best_index": result.best_index,
            "best_candidate_objective": result.best_candidate_objective.flatten().tolist(),
            "best_candidate_decision": result.best_candidate_decision.flatten().tolist(),
            "best_candidate_residual": result.best_candidate_residual,
            "metadata": result.metadata,
        }
