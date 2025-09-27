import click

from ...application.datasets.generate_dataset.generate_dataset_command import (
    AlgorithmType,
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateDatasetCommand,
    OptimizerType,
    ProblemType,
)
from ...application.datasets.generate_dataset.generate_dataset_handler import (
    GenerateDatasetCommandHandler,
)
from ...application.datasets.factories.algorithm import AlgorithmFactory
from ...application.datasets.factories.optimizer import OptimizerFactory
from ...application.datasets.factories.problem import ProblemFactory
from ...infrastructure.datasets.repositories.generated_dataset_repo import (
    FileSystemGeneratedDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option(
    "--problem-id", required=True, type=int, help='Pareto problem ID (e.g., "55", "59")'
)
def generate_data(problem_id: int):
    problem_config = ApplicationProblemConfig(
        problem_id=problem_id, type=ProblemType.biobj
    )
    algorithm_config = ApplicationAlgorithmConfig(
        type=AlgorithmType.nsga2, population_size=500
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
    command = GenerateDatasetCommand(
        problem_config=problem_config,
        algorithm_config=algorithm_config,
        optimizer_config=optimizer_config,
    )

    # Setup dependencies (could later be moved to a container or bootstrap file)
    handler = GenerateDatasetCommandHandler(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        data_model_repository=FileSystemGeneratedDatasetRepository(),
        logger=CMDLogger(),
    )

    # Execute
    handler.execute(command)


if __name__ == "__main__":
    generate_data()
