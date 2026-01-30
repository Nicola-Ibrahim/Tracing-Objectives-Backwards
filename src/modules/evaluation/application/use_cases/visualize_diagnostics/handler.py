from .....shared.domain.interfaces.base_logger import BaseLogger
from ....domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .command import VisualizeInverseEstimatorDiagnosticCommand


class VisualizeInverseEstimatorDiagnosticCommandHandler:
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
        self._diag_repo = diagnostic_repository
        self._visualizer = visualizer
        self._logger = logger

    def execute(self, command: VisualizeInverseEstimatorDiagnosticCommand) -> None:
        self._logger.log_info(
            f"Loading diagnostic results for visualization on '{command.dataset_name}'..."
        )

        # 1. Fetch requested runs using repository batch logic
        try:
            results_map_entities = self._diag_repo.get_batch(
                requests=command.requests,
                dataset_name=command.dataset_name,
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
