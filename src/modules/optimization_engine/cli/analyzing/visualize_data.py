from ...application.analysis.visualize_biobj_data.visualize_biobj_data_command import (
    VisualizeBiobjDataCommand,
)
from ...application.analysis.visualize_biobj_data.visualize_biobj_data_handler import (
    VisulizeBiobjDataCommandHandler,
)
from ...application.factories.normalizer import NormalizerFactory
from ...application.model_management.train_model.train_model_command import (
    NormalizerConfig,
)
from ...infrastructure.repositories.generation.data_model_repo import (
    FileSystemDataModelRepository,
)
from ...infrastructure.visualizers.pareto_data import PlotlyParetoDataVisualizer


def analyze_data():
    command = VisualizeBiobjDataCommand(
        data_file_name="pareto_data",
        normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
    )
    handler = VisulizeBiobjDataCommandHandler(
        data_repo=FileSystemDataModelRepository(),
        normalizer_factory=NormalizerFactory(),
        visualizer=PlotlyParetoDataVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
