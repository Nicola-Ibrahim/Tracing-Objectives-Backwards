import click

from ...modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
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
    Real-time step: Generate physically coherent design candidates for a specific target objective.
    """
    logger = CMDLogger(name="GenerateCoherentCandidatesLogger")
    domain_service = CandidateGenerationDomainService()

    config = GenerationConfig(
        dataset_name=dataset_name,
        target_objective=[400, 1600],
        n_samples=10,
        concentration_factor=10.0,
        trust_radius=0.05,
        error_threshold=None,
    )
    service = GenerateCoherentCandidatesService(
        context_repository=FileSystemContextRepository(),
        generation_domain_service=domain_service,
        logger=logger,
    )

    service.execute(config)


if __name__ == "__main__":
    main()
