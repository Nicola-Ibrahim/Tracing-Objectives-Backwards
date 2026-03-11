import click

from ...modules.dataset.application.generation import (
    DatasetConfiguration,
    GenerateDatasetService,
)
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
    config = DatasetConfiguration(
        problem_id=function_id,
        type="biobj",
        n_var=2,
        population_size=70,
        generations=10,
        save_history=True,
        dataset_name=f"cocoex_f{function_id}",
    )

    service = GenerateDatasetService(
        data_model_repository=FileSystemDatasetRepository(),
        logger=CMDLogger(),
    )

    service.execute(config)


if __name__ == "__main__":
    generate_data()
