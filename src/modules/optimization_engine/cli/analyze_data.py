from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataCommandHandler,
)
from ..domain.services.pareto_data_service import ParetoDataService
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.visualizers.mapper import ParetoVisualizationMapper
from ..infrastructure.visualizers.plotly import PlotlyParetoVisualizer


def analyze_data():
    # --- Use the correctly instantiated service in the ACL ---
    command = AnalyzeBiobjDataCommand()
    handler = AnalyzeBiobjDataCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        pareto_data_service=ParetoDataService(),
        visualizer=PlotlyParetoVisualizer(),
        pareto_data_mapper=ParetoVisualizationMapper(),
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
