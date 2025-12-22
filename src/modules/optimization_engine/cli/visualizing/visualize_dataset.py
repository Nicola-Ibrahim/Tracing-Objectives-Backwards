from ...application.visualizing.visualize_dataset.command import (
    VisualizeDatasetCommand,
)
from ...application.visualizing.visualize_dataset.handler import (
    VisualizeDatasetCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.visualization.datasets.visualizer import PlotlyDatasetVisualizer


def main():
    command = VisualizeDatasetCommand(
        data_file_name="dataset", processed_file_name="dataset"
    )
    handler = VisualizeDatasetCommandHandler(
        dataset_repo=FileSystemDatasetRepository(),
        visualizer=PlotlyDatasetVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
