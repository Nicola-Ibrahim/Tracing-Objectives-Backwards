from ...application.visualization.visualize_biobj_data.visualize_biobj_data_command import (
    VisualizeBiobjDataCommand,
)
from ...application.visualization.visualize_biobj_data.visualize_biobj_data_handler import (
    VisulizeBiobjDataCommandHandler,
)
from ...application.factories.normalizer import NormalizerFactory
from ...application.modeling.train_model.train_model_command import (
    NormalizerConfig,
)
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.visualizers.pareto_data import PlotlyParetoDataVisualizer


def analyze_data():
    command = VisualizeBiobjDataCommand(
        data_file_name="dataset",
        normalizer_config=NormalizerConfig(
            type="MinMaxScaler", params={"feature_range": (0, 1)}
        ),
    )
    handler = VisulizeBiobjDataCommandHandler(
        data_repo=FileSystemGeneratedDatasetRepository(),
        normalizer_factory=NormalizerFactory(),
        visualizer=PlotlyParetoDataVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    analyze_data()
