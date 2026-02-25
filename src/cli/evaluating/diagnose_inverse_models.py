import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases import (
    DiagnoseInverseModelsParams,
    DiagnoseInverseModelsService,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.domain.services.preprocessing_service import (
    PreprocessingService,
)
from ...modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Run diagnostic computation and persist results to the auditor")
def cli():
    candidates = [
        InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=9),
    ]

    params = DiagnoseInverseModelsParams(
        dataset_name="cocoex_f5",
        inverse_estimator_candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
        num_samples=500,
        random_state=42,
        scale_method="iqr",
    )

    service = DiagnoseInverseModelsService(
        data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemTrainedPipelineRepository(),
        diagnostic_repository=FileSystemDiagnosticRepository(),
        logger=CMDLogger(name="DiagnoseInverseModelsLogger"),
        preprocessing_service=PreprocessingService(),
    )

    service.execute(params)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
