from ...application.analysis.visualize_biobj_data.visualize_biobj_data_command import (
    VisualizeBiobjDataCommand,
)
from ...application.analysis.visualize_biobj_data.visualize_biobj_data_handler import (
    VisulizeBiobjDataCommandHandler,
)
from ...infrastructure.repositories.generation.data_model_repo import (
    FileSystemDataModelRepository,
)
from ...infrastructure.visualizers.pareto_data import PlotlyParetoDataVisualizer


def analyze_data():
    command = VisualizeBiobjDataCommand()
    handler = VisulizeBiobjDataCommandHandler(
        pareto_data_repo=FileSystemDataModelRepository(),
        visualizer=PlotlyParetoDataVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
