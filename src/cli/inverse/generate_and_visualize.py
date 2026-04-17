import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.generation.application.visualize_generation_usecase import (
    VisualizeGenerationUseCase,
)
from ...modules.generation.domain.config import GenerationConfig
from ...modules.generation.domain.services.generation import (
    CandidateGenerationDomainService,
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
    Generate candidates for a target objective and visualize them against the context.
    Uses the new decoupled architecture.
    """
    logger = CMDLogger(name="GenerateAndVisualizeCli")

    # Use Case Orchestrator
    use_case = VisualizeGenerationUseCase(
        context_repository=FileSystemContextRepository(),
        generation_domain_service=CandidateGenerationDomainService(),
        dataset_repository=FileSystemDatasetRepository(),
        visualizer=PlotlyContextVisualizer(),
        logger=logger,
    )

    config = GenerationConfig(
        dataset_name=dataset_name,
        target_objective=[400, 1600],
        n_samples=20,
        concentration_factor=10.0,
        trust_radius=0.05,
        error_threshold=None,
    )

    use_case.execute(config)


if __name__ == "__main__":
    main()
