import click

from ...application.assuring.select_inverse_model import (
    ModelCandidate,
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

    # Define candidates (types and optional versions)
    # Example: Compare latest MDN vs latest CVAE
    candidates = [
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=1),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=2),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=3),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=4),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=5),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version=6),
        ModelCandidate(type=EstimatorTypeEnum.CVAE, version=1),
    ]

    command = SelectInverseModelCommand(
        candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
