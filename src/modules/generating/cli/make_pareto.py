import click

from ...shared.adapters.archivers.npz import ParetoNPzArchiver
from ..infrastructure.algorithms import AlgorithmFactory
from ..infrastructure.optimizers import OptimizerFactory
from ..infrastructure.problems import ProblemFactory
from ..application.generate_biobj_pareto_data.generate_biobj_pareto_data_handler import (
    GenerateBiobjParetoDataHandler,
)
from ..application.generate_biobj_pareto_data.generate_pareto_command import (
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateParetoCommand,
)


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
    handler = GenerateBiobjParetoDataHandler(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        archiver=ParetoNPzArchiver(),
    )

    # 3. Execute
    output_path = handler.execute(command)
    click.echo(f"Pareto data saved to: {output_path}")


if __name__ == "__main__":
    generate_data()
