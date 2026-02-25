import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases import (
    CheckModelPerformanceParams,
    CheckModelPerformanceService,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.infrastructure.visualization.model_performance_2d.visualizer import (
    ModelPerformance2DVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.domain.services.preprocessing_service import (
    PreprocessingService,
)
from ...modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)


@click.command(help="Visualize a trained model's performance diagnostics")
def main() -> None:
    service = CheckModelPerformanceService(
        model_repository=FileSystemTrainedPipelineRepository(),
        data_repository=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
        preprocessing_service=PreprocessingService(),
    )
    params = CheckModelPerformanceParams(
        dataset_name="cocoex_f5",
        estimator=InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=11),
        n_samples=50,
    )
    service.execute(params)


if __name__ == "__main__":
    main()
