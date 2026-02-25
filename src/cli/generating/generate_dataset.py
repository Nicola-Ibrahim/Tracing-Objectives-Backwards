import click

from ...modules.dataset.application.factories.algorithm import AlgorithmFactory
from ...modules.dataset.application.factories.optimizer import OptimizerFactory
from ...modules.dataset.application.factories.problem import ProblemFactory
from ...modules.dataset.application.generation import (
    AlgorithmType,
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateDatasetParams,
    GenerateDatasetService,
    OptimizerType,
    ProblemType,
)
from ...modules.dataset.domain.services import DatasetGenerationService
from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option(
    "--function-id",
    "--problem-id",
    required=True,
    type=int,
    help='COCO function index (e.g., "5", "55")',
)
def generate_data(function_id: int):
    problem_config = ApplicationProblemConfig(
        problem_id=function_id, type=ProblemType.biobj
    )
    algorithm_config = ApplicationAlgorithmConfig(
        type=AlgorithmType.nsga2, population_size=500
    )
    optimizer_config = ApplicationOptimizerConfig(
        type=OptimizerType.minimizer,
        generations=16,
        save_history=True,
        verbose=False,
    )

    # Build params from CLI arguments
    params = GenerateDatasetParams(
        problem_config=problem_config,
        algorithm_config=algorithm_config,
        optimizer_config=optimizer_config,
        dataset_name=f"cocoex_f{function_id}",
    )

    # Setup dependencies (could later be moved to a container or bootstrap file)
    logger = CMDLogger()
    dataset_service = DatasetGenerationService()
    service = GenerateDatasetService(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        data_model_repository=FileSystemDatasetRepository(),
        dataset_service=dataset_service,
        logger=logger,
    )

    # Execute
    service.execute(params)


if __name__ == "__main__":
    generate_data()
