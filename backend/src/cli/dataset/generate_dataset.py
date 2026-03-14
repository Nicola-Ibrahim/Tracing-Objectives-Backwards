import click

from ...modules.dataset.application.dataset_service import (
    DatasetConfiguration,
    DatasetService,
)
from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ...modules.dataset.infrastructure.sources.factory import DataGeneratorFactory
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command()
@click.option(
    "--function-id",
    "--problem-id",
    required=True,
    type=int,
    help='COCO function index (e.g., "5", "55")',
)
@click.option(
    "--n-var",
    type=int,
    default=2,
    help="Number of decision variables",
)
def generate_data(function_id: int, n_var: int):
    config = DatasetConfiguration(
        dataset_name=f"cocoex_f{function_id}_d{n_var}",
        generator_type="coco_pymoo",
        params={
            "n_var": n_var,
            "problem_id": function_id,
            "generations": 10,
            "population_size": 70,
        },
        split_ratio=0.2,
        random_state=42
    )

    logger = CMDLogger(name="DataCLI")
    service = DatasetService(
        repository=FileSystemDatasetRepository(),
        engine_repository=FileSystemInverseMappingEngineRepository(),
        generator_factory=DataGeneratorFactory(),
        logger=logger,
    )

    result = service.generate_dataset(config)
    if result.is_ok:
        print(f"Dataset generated successfully: {result.value}")
    else:
        print(f"Generation failed: {result.error.message}")


if __name__ == "__main__":
    generate_data()
