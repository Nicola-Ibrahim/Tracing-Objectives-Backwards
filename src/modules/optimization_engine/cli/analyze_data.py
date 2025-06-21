from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataCommandHandler,
)
from ..domain.services.pareto_data_service import ParetoDataService
from ..infrastructure.archivers.npz import ParetoNPzArchiver
from ..infrastructure.visualizers.plotly import PlotlyParetoVisualizer
from ..infrastructure.visualizers.mapper import ParetoVisualizationMapper

def analyze_data():
    # --- Use the correctly instantiated service in the ACL ---
    command = AnalyzeBiobjDataCommand()
    handler = AnalyzeBiobjDataCommandHandler(
        pareto_data_service=ParetoDataService(archiver=ParetoNPzArchiver()),
        visualizer=PlotlyParetoVisualizer(),
        pareto_data_mapper=ParetoVisualizationMapper()
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
