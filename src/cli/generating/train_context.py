import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.generation.application.train_context import (
    TrainContextParams,
    TrainContextService,
)
from ...modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option("--dataset-name", required=True, type=str, help="Name of the dataset")
def main(dataset_name: str):
    """
    Offline step: Prepare the coherence context (Delaunay mesh, coherence threshold) for a dataset.
    """
    logger = CMDLogger(name="PrepareContextLogger")

    params = TrainContextParams(
        dataset_name=dataset_name,
        k_neighbors=5,
        transforms=[
            {"type": "min_max", "target": "decisions"},
            {"type": "min_max", "target": "objectives"},
        ],
    )

    service = TrainContextService(
        dataset_repository=FileSystemDatasetRepository(),
        context_repository=FileSystemContextRepository(),
        logger=logger,
    )
    service.execute(params)


if __name__ == "__main__":
    main()
