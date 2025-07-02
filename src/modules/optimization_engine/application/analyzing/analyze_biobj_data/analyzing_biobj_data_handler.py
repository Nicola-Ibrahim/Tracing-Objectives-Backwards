from ....domain.analyzing.interfaces.base_visualizer import (
    BaseDataVisualizer,
)
from ....domain.generation.interfaces.base_repository import (
    BaseParetoDataRepository,
)
from .analyzing_biobj_data_command import AnalyzeBiobjDataCommand


class AnalyzeBiobjDataCommandHandler:
    """
    Handles the AnalyzeBiobjDataCommand by using the ParetoDataService
    to retrieve data and then visualizing it.
    """

    def __init__(
        self,
        pareto_data_repo: BaseParetoDataRepository,
        visualizer: BaseDataVisualizer,
    ):
        """
        Initializes the command handler with a dedicated analysis data service and a visualizer.

        Args:
            pareto_data_service: The service responsible for fetching analysis-ready data.
            visualizer: The component responsible for plotting the data.
        """
        self._pareto_data_repo = pareto_data_repo
        self._visualizer = visualizer

    def execute(self, command: AnalyzeBiobjDataCommand) -> None:
        """
        Executes the command to analyze and visualize biobjective data.

        It delegates data retrieval to the ParetoDataService and then
        passes the received data to the visualizer.

        Args:
            command: The command containing the identifier for the data to analyze.
        """
        # Retrieve the dataset using the repository
        pareto_data = self._pareto_data_repo.load(filename=command.data_file_name)

        # Visualize the data using the provided visualizer
        self._visualizer.plot(data=pareto_data)
