import click

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


@click.command(help="Visualize a processed dataset")
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to visualize.",
)
def main(dataset_name: str):
    command = VisualizeDatasetCommand(dataset_name=dataset_name)
    handler = VisualizeDatasetCommandHandler(
        dataset_repo=FileSystemDatasetRepository(),
        visualizer=PlotlyDatasetVisualizer(),
    )

    handler.execute(command)


if __name__ == "__main__":
    main()
