import click

from ...dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from ...shared.infrastructure.loggers.cmd_logger import CMDLogger
from ..application.use_cases.prepare_context import PrepareContextService
from ..infrastructure.repositories.context_repo import FileSystemContextRepository


@click.command()
@click.option("--dataset-name", required=True, type=str, help="Name of the dataset")
@click.option(
    "--surrogate-type",
    type=str,
    default="rbf",
    help="Type of the forward surrogate estimator",
)
@click.option(
    "--k-neighbors",
    type=int,
    default=5,
    help="Number of neighbors for coherence threshold calculation",
)
def main(dataset_name: str, surrogate_type: str, k_neighbors: int):
    """
    Offline step: Prepare the coherence context (Delaunay mesh, coherence threshold) for a dataset.
    """
    logger = CMDLogger(name="PrepareContextLogger")
    service = PrepareContextService(
        dataset_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemTrainedPipelineRepository(),
        context_repository=FileSystemContextRepository(),
        logger=logger,
    )
    service.execute(dataset_name, surrogate_type, k_neighbors)


if __name__ == "__main__":
    main()
