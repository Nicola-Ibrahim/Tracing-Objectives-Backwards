import click

from ...application.factories.algorithm import AlgorithmFactory
from ...application.factories.normalizer import NormalizerFactory
from ...application.factories.optimizer import OptimizerFactory
from ...application.factories.problem import ProblemFactory
from ...application.generating.generate_dataset.command import (
    AlgorithmType,
    ApplicationAlgorithmConfig,
    ApplicationOptimizerConfig,
    ApplicationProblemConfig,
    GenerateDatasetCommand,
    OptimizerType,
    ProblemType,
)
from ...domain.modeling.enums.normalizer_type import NormalizerTypeEnum
from ...domain.modeling.value_objects.estimator_params import NormalizerConfig
from ...application.generating.generate_dataset.handler import (
    GenerateDatasetCommandHandler,
)
from ...domain.datasets.services import DatasetGenerationService
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger


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

    # Build command from CLI arguments
    command = GenerateDatasetCommand(
        problem_config=problem_config,
        algorithm_config=algorithm_config,
        optimizer_config=optimizer_config,
        dataset_name=f"cocoex_f{function_id}",
        normalizer_config=NormalizerConfig(
            type=NormalizerTypeEnum.HYPERCUBE, params={}
        ),
        test_size=0.2,
        random_state=42,
    )

    # Setup dependencies (could later be moved to a container or bootstrap file)
    logger = CMDLogger()
    dataset_service = DatasetGenerationService()
    handler = GenerateDatasetCommandHandler(
        problem_factory=ProblemFactory(),
        algorithm_factory=AlgorithmFactory(),
        optimizer_factory=OptimizerFactory(),
        data_model_repository=FileSystemDatasetRepository(),
        dataset_service=dataset_service,
        normalizer_factory=NormalizerFactory(),
        logger=logger,
    )

    # Execute
    handler.execute(command)


if __name__ == "__main__":
    generate_data()
