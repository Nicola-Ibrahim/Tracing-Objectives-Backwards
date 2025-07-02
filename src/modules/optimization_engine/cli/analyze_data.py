from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_command import (
    AnalyzeBiobjDataCommand,
)
from ..application.analyzing.analyze_biobj_data.analyzing_biobj_data_handler import (
    AnalyzeBiobjDataCommandHandler,
)
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)
from ..infrastructure.visualizers.pareto_data import PlotlyParetoDataVisualizer


def analyze_data():
    # --- Use the correctly instantiated service in the ACL ---
    command = AnalyzeBiobjDataCommand()
    handler = AnalyzeBiobjDataCommandHandler(
        pareto_data_repo=NPZParetoDataRepository(),
        visualizer=PlotlyParetoDataVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
