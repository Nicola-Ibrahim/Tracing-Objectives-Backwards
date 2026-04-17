import click

from ...modules.inverse.application.generate_candidates import (
    GeneratCandidatesService,
    GenerationConfig,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option("--dataset_name", required=True, type=str, help="Name of the dataset")
@click.option("--solver_type", default="GBPI", help="Type of inverse solver to use")
@click.option(
    "--version", default=None, type=int, help="Optional engine version number"
)
def main(dataset_name: str, solver_type: str, version: int | None):
    """
    Real-time step: Generate physically coherent design candidates for a
    specific target objective.
    """
    logger = CMDLogger(name="GenerateCoherentCandidatesLogger")

    config = GenerationConfig(
        dataset_name=dataset_name,
        target_objective=[400, 1600],
        n_samples=10,
        concentration_factor=10.0,
        trust_radius=0.05,
        solver_type=solver_type,
        version=version,
    )
    service = GeneratCandidatesService(
        inverse_mapping_engine_repository=FileSystemInverseMappingEngineRepository(),
        logger=logger,
    )

    service.execute(config)


if __name__ == "__main__":
    main()
