from ...application.visualization.visualize_dataset.visualize_data_command import (
    VisualizeDatasetCommand,
)
from ...application.visualization.visualize_dataset.visualize_data_handler import (
    VisualizeDatasetCommandHandler,
)
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.repositories.datasets.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.visualizers.pareto_data import PlotlyDatasetVisualizer

command = VisualizeDatasetCommand(
    data_file_name="dataset", processed_file_name="dataset"
)
handler = VisualizeDatasetCommandHandler(
    dataset_repo=FileSystemGeneratedDatasetRepository(),
    processed_dataset_repo=FileSystemProcessedDatasetRepository(),
    visualizer=PlotlyDatasetVisualizer(),
)

handler.execute(command)
