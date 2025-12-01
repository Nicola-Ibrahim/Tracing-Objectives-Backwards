import click

from ...application.factories.estimator import EstimatorFactory
from ...application.assurance.inverse_model_validation import (
    ValidateInverseModelCommand,
    ValidateInverseModelHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum

@click.command(help="Validate inverse model using forward simulator")
def cli():
    handler = ValidateInverseModelHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ValidationCMDLogger"),
        estimator_factory=EstimatorFactory(),
    )

    command = ValidateInverseModelCommand(
        inverse_estimator_type=EstimatorTypeEnum.MDN,
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)



def main() -> None:
    cli()


if __name__ == "__main__":
    main()
