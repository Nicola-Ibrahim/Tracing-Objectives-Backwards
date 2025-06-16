from ...domain.interfaces.base_visualizer import BaseParetoVisualizer
from ...domain.service.anaylzing_data_service import BiobjAnalysisDataService
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataCommandHandler:
    """
    Handles the AnalyzeBiobjDataCommand by using the BiobjAnalysisDataService
    to retrieve data and then visualizing it.
    """

    def __init__(
        self,
        analysis_data_service: BiobjAnalysisDataService,
        visualizer: BaseParetoVisualizer,
    ):
        """
        Initializes the command handler with a dedicated analysis data service and a visualizer.

        Args:
            analysis_data_service: The service responsible for fetching analysis-ready data.
            visualizer: The component responsible for plotting the data.
        """
        self._analysis_data_service = analysis_data_service
        self._visualizer = visualizer

    def execute(self, command: AnalyzeBiobjDataCommand) -> None:
        """
        Executes the command to analyze and visualize biobjective data.

        It delegates data retrieval to the BiobjAnalysisDataService and then
        passes the received data to the visualizer.

        Args:
            command: The command containing the identifier for the data to analyze.
        """
        # The command handler now depends on the BiobjAnalysisDataService
        analysis_result = self._analysis_data_service.get_biobj_analysis_data(
            data_identifier=command.filename
        )

        # The visualizer still receives the data in the analyzing context's
        # preferred format (AnalysisResultDTO components)
        self._visualizer.plot(
            pareto_set=analysis_result.solutions,
            pareto_front=analysis_result.objectives,
        )
