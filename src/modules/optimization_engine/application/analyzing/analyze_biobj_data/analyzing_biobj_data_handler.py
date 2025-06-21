from ....domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer
from ....domain.services.pareto_data_service import ParetoDataService
from ....infrastructure.visualizers.mapper import ParetoVisualizationMapper
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataCommandHandler:
    """
    Handles the AnalyzeBiobjDataCommand by using the ParetoDataService
    to retrieve data and then visualizing it.
    """

    def __init__(
        self,
        pareto_data_service: ParetoDataService,
        visualizer: BaseParetoVisualizer,
        pareto_data_mapper: ParetoVisualizationMapper,
    ):
        """
        Initializes the command handler with a dedicated analysis data service and a visualizer.

        Args:
            pareto_data_service: The service responsible for fetching analysis-ready data.
            visualizer: The component responsible for plotting the data.
        """
        self._pareto_data_service = pareto_data_service
        self._visualizer = visualizer
        self._pareto_data_mapper = pareto_data_mapper

    def execute(self, command: AnalyzeBiobjDataCommand) -> None:
        """
        Executes the command to analyze and visualize biobjective data.

        It delegates data retrieval to the ParetoDataService and then
        passes the received data to the visualizer.

        Args:
            command: The command containing the identifier for the data to analyze.
        """
        f1_data, interp_data = self._pareto_data_service.provide_visualization_data(
            data_identifier=command.filename
        )

        # Map the raw data to a structured DTO
        dto = self._pareto_data_mapper.map_to_dto(f1_data, interp_data)

        # Visualize the data using the provided visualizer
        self._visualizer.plot(dto=dto)
