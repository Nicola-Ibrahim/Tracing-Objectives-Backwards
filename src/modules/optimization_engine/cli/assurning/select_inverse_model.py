import click

from ...application.assuring.select_inverse_model import (
    SelectInverseModelCommand,
    SelectInverseModelHandler,
)
from ...application.factories.estimator import EstimatorFactory
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


@click.command(help="Select the best inverse model by comparing multiple candidates")
def cli():
    handler = SelectInverseModelHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ModelSelectionLogger"),
        estimator_factory=EstimatorFactory(),
    )

    # Hardcoded list of models to comparison as per request
    estimator_types = [
        EstimatorTypeEnum.MDN,
        EstimatorTypeEnum.CVAE,
        # EstimatorTypeEnum.FLOW
    ]

    command = SelectInverseModelCommand(
        inverse_estimator_types=estimator_types,
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
