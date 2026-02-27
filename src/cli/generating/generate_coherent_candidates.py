import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
    GenerationConfig,
)
from ...modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)
from ...modules.generation.infrastructure.visualization.visualizer import (
    PlotlyContextVisualizer,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option("--dataset-name", required=True, type=str, help="Name of the dataset")
def main(
    dataset_name: str,
):
    """
    Real-time step: Generate physically coherent design candidates for a specific target objective.
    """
    logger = CMDLogger(name="GenerateCoherentCandidatesLogger")
    service = GenerateCoherentCandidatesService(
        context_repository=FileSystemContextRepository(),
        dataset_repository=FileSystemDatasetRepository(),
        logger=logger,
        visualizer=PlotlyContextVisualizer(),
    )

    config = GenerationConfig(
        dataset_name=dataset_name,
        target_objective=[400, 1600],
        n_samples=10,
        concentration_factor=10.0,
        trust_radius=0.05,
        error_threshold=None,
    )

    service.execute(config)


if __name__ == "__main__":
    main()
