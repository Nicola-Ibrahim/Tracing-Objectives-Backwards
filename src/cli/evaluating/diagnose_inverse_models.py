import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.diagnose_models import (
    DiagnoseInverseModelsParams,
    DiagnoseInverseModelsService,
    InverseEngineCandidate,
)
from ...modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Run diagnostic computation and persist results to the auditor")
def cli():
    # Example using the new engine repository
    candidates = [
        InverseEngineCandidate(solver_type="GBPI", version=1),
    ]

    params = DiagnoseInverseModelsParams(
        dataset_name="cocoex_f5",
        inverse_engine_candidates=candidates,
        num_samples=200,
        random_state=42,
        scale_method="sd",
    )

    service = DiagnoseInverseModelsService(
        data_repository=FileSystemDatasetRepository(),
        engine_repository=FileSystemInverseMappingEngineRepository(),
        diagnostic_repository=FileSystemDiagnosticRepository(),
        logger=CMDLogger(name="DiagnoseInverseModelsLogger"),
    )

    service.execute(params)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
