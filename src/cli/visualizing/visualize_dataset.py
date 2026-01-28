import click

from ...modules.dataset.application.visualizing import (
    VisualizeDatasetCommand,
    VisualizeDatasetCommandHandler,
)
from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.dataset.infrastructure.visualization import (
    PlotlyDatasetVisualizer,
)


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
