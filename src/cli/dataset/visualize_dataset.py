import click

from ...modules.dataset.application.visualization import (
    VisualizeDatasetParams,
    VisualizeDatasetService,
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
    params = VisualizeDatasetParams(dataset_name=dataset_name)
    service = VisualizeDatasetService(
        dataset_repo=FileSystemDatasetRepository(),
        visualizer=PlotlyDatasetVisualizer(),
    )

    service.execute(params)


if __name__ == "__main__":
    main()
