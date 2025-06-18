import click

from ...application.generate_biobj_pareto_data.generate_biobj_pareto_data_handler import (
    GenerateBiobjParetoDataCommandHandler,
)
from ...application.generate_biobj_pareto_data.generate_pareto_command import (
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateParetoCommand,
)
from ...domain.services.pareto_generation_service import ParetoGenerationService
from ...infrastructure.algorithms import AlgorithmFactory
from ...infrastructure.archivers.npz import ParetoNPzArchiver
from ...infrastructure.optimizers import OptimizerFactory
from ...infrastructure.problems import ProblemFactory


@click.command()
@click.option(
    "--problem-id", required=True, type=int, help='Pareto problem ID (e.g., "55", "59")'
)
def generate_data(problem_id: int):
    # 1. Build command from CLI arguments
    command = GenerateParetoCommand(
        problem_config=ApplicationProblemConfig(id=problem_id),
        algorithm_config=ApplicationAlgorithmConfig(),  # Uses default NSGA2
        optimizer_config=ApplicationOptimizerConfig(),  # Uses default Minimizer
    )

    # 2. Setup dependencies (could later be moved to a container or bootstrap file)
    handler = GenerateBiobjParetoDataCommandHandler(
        ParetoGenerationService(
            problem_factory=ProblemFactory(),
            algorithm_factory=AlgorithmFactory(),
            optimizer_factory=OptimizerFactory(),
            archiver=ParetoNPzArchiver(),
        )
    )

    # 3. Execute
    output_path = handler.execute(command)
    click.echo(f"Pareto data saved to: {output_path}")


if __name__ == "__main__":
    generate_data()
