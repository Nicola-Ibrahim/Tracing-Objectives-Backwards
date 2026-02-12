import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases.diagnose_inverse_models import (
    DiagnoseInverseModelsCommand,
    DiagnoseInverseModelsHandler,
    InverseEstimatorCandidate,
)
from ...modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Run diagnostic computation and persist results to the auditor")
def cli():
    candidates = [
        InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=4),
    ]

    command = DiagnoseInverseModelsCommand(
        dataset_name="cocoex_f5",
        inverse_estimator_candidates=candidates,
        forward_estimator_type=EstimatorTypeEnum.COCO,
        num_samples=500,
        random_state=42,
        scale_method="iqr",
    )

    handler = DiagnoseInverseModelsHandler(
        data_repository=FileSystemDatasetRepository(),
        model_artifact_repository=FileSystemModelArtifactRepository(),
        diagnostic_repository=FileSystemDiagnosticRepository(),
        logger=CMDLogger(name="DiagnoseInverseModelsLogger"),
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
