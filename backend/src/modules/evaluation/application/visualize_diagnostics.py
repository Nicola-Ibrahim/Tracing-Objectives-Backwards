from pydantic import BaseModel, Field

from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ..domain.interfaces.base_visualizer import BaseVisualizer


class InverseEngineDiagnosticCandidate(BaseModel):
    """Identifies a specific inverse engine run to visualize."""

    solver_type: str = Field(..., description="The type of solver evaluated.")
    version: int = Field(..., description="The specific version of the engine.")
    run_number: int | None = Field(
        default=None,
        description="The sequential run ID (1, 2...). If None, latest run is loaded.",
    )


class VisualizeInverseEstimatorDiagnosticParams(BaseModel):
    """Command payload for visualizing inverse engine diagnostic results."""

    dataset_name: str = Field(
        ...,
        description="The dataset context the diagnostics were run on.",
        examples=["cocoex_f5"],
    )

    inverse_engine_candidates: list[InverseEngineDiagnosticCandidate] = Field(
        ...,
        description="List of specific inverse engine runs to load and compare.",
    )


class VisualizeInverseEstimatorDiagnosticService:
    """
    Orchestrator for loading and rendering previously computed diagnostics.
    Enables re-visualizing any historical evaluation run.
    """

    def __init__(
        self,
        diagnostic_repository: BaseDiagnosticRepository,
        visualizer: BaseVisualizer,
        logger: BaseLogger,
    ):
        self._diagnostic_repository = diagnostic_repository
        self._visualizer = visualizer
        self._logger = logger

    def execute(self, params: VisualizeInverseEstimatorDiagnosticParams) -> None:
        self._logger.log_info(
            f"Loading diagnostic results for '{params.dataset_name}'..."
        )

        # 1. Fetch requested runs using repository batch logic
        try:
            results_map_entities = self._diagnostic_repository.get_batch(
                estimators=params.inverse_engine_candidates,
                dataset_name=params.dataset_name,
            )
        except FileNotFoundError as e:
            self._logger.log_error(f"Failed to load diagnostic results: {e}")
            raise

        # 2. Generate plots
        if results_map_entities:
            results_list = list(results_map_entities.values())
            self._logger.log_info(f"Rendering plots for {len(results_list)} models...")
            self._visualizer.plot(results_list)
        else:
            self._logger.log_warning("No diagnostic results found to visualize.")
