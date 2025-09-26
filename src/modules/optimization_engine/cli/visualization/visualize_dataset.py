from ...application.visualization.visualize_dataset.visualize_data_command import (
    VisualizeDatasetCommand,
)
from ...application.visualization.visualize_dataset.visualize_data_handler import (
    VisualizeDatasetCommandHandler,
)
from ...infrastructure.datasets.repositories.processed_dataset_repo import (
    FileSystemProcessedDatasetRepository,
)
from ...infrastructure.datasets.visualizers.dataset import PlotlyDatasetVisualizer
from ...infrastructure.repositories.datasets.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)

command = VisualizeDatasetCommand(
    data_file_name="dataset", processed_file_name="dataset"
)
handler = VisualizeDatasetCommandHandler(
    dataset_repo=FileSystemGeneratedDatasetRepository(),
    processed_dataset_repo=FileSystemProcessedDatasetRepository(),
    visualizer=PlotlyDatasetVisualizer(),
)

handler.execute(command)
