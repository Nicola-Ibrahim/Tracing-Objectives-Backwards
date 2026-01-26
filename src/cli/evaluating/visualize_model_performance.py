import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases.check_model_performance import (
    CheckModelPerformanceCommand,
    CheckModelPerformanceCommandHandler,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.infrastructure.visualization.model_performance_2d.visualizer import (
    ModelPerformance2DVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


@click.command(help="Visualize a trained model's performance diagnostics")
def main() -> None:
    handler = CheckModelPerformanceCommandHandler(
        model_artificat_repo=FileSystemModelArtifactRepository(),
        processed_dataset_repo=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
    )
    command = CheckModelPerformanceCommand(
        dataset_name="cocoex_f5",
        estimator=InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=11),
        n_samples=50,
    )
    handler.execute(command)


if __name__ == "__main__":
    main()
