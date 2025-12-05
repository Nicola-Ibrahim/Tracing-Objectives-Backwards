import click

from ...application.assuring.inverse_model_validation import (
    ValidateInverseModelCommand,
    ValidateInverseModelHandler,
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


@click.command(help="Validate inverse model using forward simulator")
@click.option(
    "--estimator",
    type=click.Choice([e.value for e in EstimatorTypeEnum]),
    default=EstimatorTypeEnum.MDN.value,
    show_default=True,
    help="Inverse estimator type to validate.",
)
def cli(estimator: str):
    handler = ValidateInverseModelHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ValidationCMDLogger"),
        estimator_factory=EstimatorFactory(),
    )

    command = ValidateInverseModelCommand(
        inverse_estimator_type=EstimatorTypeEnum(estimator),
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
