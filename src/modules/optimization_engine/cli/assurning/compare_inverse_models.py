import click

from ...application.assuring.compare_inverse_models import (
    CompareInverseModelsCommand,
    CompareInverseModelsHandler,
    InverseEstimatorCandidate,
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
def cli():
    handler = CompareInverseModelsHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InverseComparisonLogger"),
        estimator_factory=EstimatorFactory(),
        visualizer=InverseComparisonVisualizer(),
    )

    candidates = [
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=1, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=2, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=3, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=4, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=5, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=6, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.MDN, version=7, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.CVAE, version=0, dataset_name="cocoex_f5"
        ),
        InverseEstimatorCandidate(
            type=EstimatorTypeEnum.CVAE, version=1, dataset_name="cocoex_f5"
        ),
    ]

    command = CompareInverseModelsCommand(
        candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
