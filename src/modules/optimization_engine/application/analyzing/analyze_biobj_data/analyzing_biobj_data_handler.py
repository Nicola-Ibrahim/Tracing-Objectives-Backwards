from ....domain.analyzing.interfaces.base_visualizer import BaseParetoVisualizer
from ....domain.services.pareto_data_service import ParetoDataService
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
    ):
        """
        Initializes the command handler with a dedicated analysis data service and a visualizer.

        Args:
            pareto_data_service: The service responsible for fetching analysis-ready data.
            visualizer: The component responsible for plotting the data.
        """
        self._pareto_data_service = pareto_data_service
        self._visualizer = visualizer

    def execute(self, command: AnalyzeBiobjDataCommand) -> None:
        """
        Executes the command to analyze and visualize biobjective data.

        It delegates data retrieval to the ParetoDataService and then
        passes the received data to the visualizer.

        Args:
            command: The command containing the identifier for the data to analyze.
        """
        f1_rel_data, x1_x2_interp_data = (
            self._pareto_data_service.provide_visualization_data(
                data_identifier=command.filename
            )
        )

        self._visualizer.plot(
            f1_rel_data=f1_rel_data,
            x1_x2_interp_data=x1_x2_interp_data,
        )
