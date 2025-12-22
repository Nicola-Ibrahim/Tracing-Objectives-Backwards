import click

from ...application.assuring.compare_inverse_models import (
    CompareInverseModelsCommand,
    CompareInverseModelsHandler,
    ModelCandidate,
)
from ...application.factories.estimator import EstimatorFactory
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger
from ...infrastructure.visualization.comparison.visualizer import (
    InverseComparisonVisualizer,
)


@click.command(help="Compare inverse model candidates against a forward simulator")
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to load for comparison.",
)
def cli(dataset_name: str):
    handler = CompareInverseModelsHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InverseComparisonLogger"),
        estimator_factory=EstimatorFactory(),
        visualizer=InverseComparisonVisualizer(),
    )

    # Define candidates (types and optional versions)
    # Example: Compare latest MDN vs latest CVAE
    candidates = [
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="1"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="2"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="3"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="4"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="5"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="6"),
        ModelCandidate(type=EstimatorTypeEnum.MDN, version="7"),
        ModelCandidate(type=EstimatorTypeEnum.CVAE, version="0"),
        ModelCandidate(type=EstimatorTypeEnum.CVAE, version="1"),
    ]

    command = CompareInverseModelsCommand(
        dataset_name=dataset_name,
        candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
