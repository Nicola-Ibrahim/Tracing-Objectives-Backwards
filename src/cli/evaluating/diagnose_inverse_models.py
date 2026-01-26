import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases.diagnose_inverse_models import (
    DiagnoseInverseModelsCommand,
    DiagnoseInverseModelsHandler,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.infrastructure.visualization.inverse_comparison.visualizer import (
    InverseComparisonVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(
    help="Run comprehensive diagnostics (Accuracy + Reliability) for inverse models"
)
def cli():
    handler = DiagnoseInverseModelsHandler(
        data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="DiagnoseInverseModelsLogger"),
        visualizer=InverseComparisonVisualizer(),
    )

    candidates = [
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=1),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=2),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=3),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=4),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=5),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=6),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=7),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=1),
        InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=1),
    ]

    command = DiagnoseInverseModelsCommand(
        dataset_name="cocoex_f5",
        candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
        num_samples=300,
        random_state=42,
        scale_method="sd",
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
