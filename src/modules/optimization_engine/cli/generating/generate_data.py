import click

from ...application.factories.algorithm import AlgorithmFactory
from ...application.factories.optimizer import OptimizerFactory
from ...application.factories.problem import ProblemFactory
from ...application.generation.generate_biobj_data.generate_biobj_data_command import (
    AlgorithmType,
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateBiobjDataCommand,
    OptimizerType,
    ProblemType,
)
from ...application.generation.generate_biobj_data.generate_biobj_data_handler import (
    GenerateBiobjDataCommandHandler,
)
from ...infrastructure.repositories.generation.data_model_repo import (
    FileSystemDataModelRepository,
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
        generations=16,
        seed=42,
        save_history=True,
        verbose=False,
        pf=False,
    )

    # Build command from CLI arguments
    command = GenerateBiobjDataCommand(
        problem_config=problem_config,
        algorithm_config=algorithm_config,
        optimizer_config=optimizer_config,
    )

    # Setup dependencies (could later be moved to a container or bootstrap file)
    handler = GenerateBiobjDataCommandHandler(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        data_model_repository=FileSystemDataModelRepository(),
    )

    # Execute
    output_path = handler.execute(command)
    click.echo(f"Pareto data saved to: {output_path}")


if __name__ == "__main__":
    generate_data()
