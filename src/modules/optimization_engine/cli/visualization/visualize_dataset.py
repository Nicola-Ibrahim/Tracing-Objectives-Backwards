from ...application.visualization.visualize_dataset.visualize_data_command import (
    VisualizeDatasetCommand,
)
from ...application.visualization.visualize_dataset.visualize_data_handler import (
    VisualizeDatasetCommandHandler,
)
from ...infrastructure.datasets.visualizers.dataset import PlotlyDatasetVisualizer
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)

command = VisualizeDatasetCommand(
    data_file_name="dataset", processed_file_name="dataset"
)
handler = VisualizeDatasetCommandHandler(
    dataset_repo=FileSystemDatasetRepository(),
    visualizer=PlotlyDatasetVisualizer(),
)

handler.execute(command)
