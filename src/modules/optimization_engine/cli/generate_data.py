import click

from ..application.paretos.generate_biobj_pareto_data.generate_biobj_pareto_data_handler import (
    GenerateBiobjParetoDataCommandHandler,
)
from ..application.paretos.generate_biobj_pareto_data.generate_pareto_command import (
    AlgorithmType,
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateParetoCommand,
    OptimizerType,
    ProblemType,
)
from ..infrastructure.algorithms import AlgorithmFactory
from ..infrastructure.optimizers import OptimizerFactory
from ..infrastructure.problems import ProblemFactory
from ..infrastructure.repositories.generation.npz_pareto_data_repo import (
    NPZParetoDataRepository,
)


@click.command()
@click.option(
    "--problem-id", required=True, type=int, help='Pareto problem ID (e.g., "55", "59")'
)
def generate_data(problem_id: int):
    problem_config = ApplicationProblemConfig(
        problem_id=problem_id, type=ProblemType.biobj
    )
    algorithm_config = ApplicationAlgorithmConfig(
        type=AlgorithmType.nsga2, population_size=200
    )
    optimizer_config = ApplicationOptimizerConfig(
        type=OptimizerType.minimizer,
        generations=100,
        seed=42,
        save_history=True,
        verbose=False,
        pf=False,
    )

    # Build command from CLI arguments
    command = GenerateParetoCommand(
        problem_config=problem_config,
        algorithm_config=algorithm_config,
        optimizer_config=optimizer_config,
    )

    # Setup dependencies (could later be moved to a container or bootstrap file)
    handler = GenerateBiobjParetoDataCommandHandler(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        archiver=NPZParetoDataRepository(),
    )

    # Execute
    output_path = handler.execute(command)
    click.echo(f"Pareto data saved to: {output_path}")


if __name__ == "__main__":
    generate_data()
