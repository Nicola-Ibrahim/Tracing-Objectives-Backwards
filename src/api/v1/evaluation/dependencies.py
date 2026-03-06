from ....modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ....modules.evaluation.application.check_engine_performance import (
    CheckModelPerformanceService,
)
from ....modules.evaluation.application.diagnose_engines import (
    DiagnoseInverseModelsService,
)
from ....modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ....modules.evaluation.infrastructure.visualization.inverse_models_comparison.visualizer import (
    InverseModelsComparisonVisualizer,
)
from ....modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ....modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_diagnose_service() -> DiagnoseInverseModelsService:
    logger = CMDLogger(name="EvaluationDiagnoseAPI")
    return DiagnoseInverseModelsService(
        engine_repository=FileSystemInverseMappingEngineRepository(),
        data_repository=FileSystemDatasetRepository(),
        diagnostic_repository=FileSystemDiagnosticRepository(),
        logger=logger,
    )


def get_performance_service() -> CheckModelPerformanceService:
    visualizer = InverseModelsComparisonVisualizer()
    return CheckModelPerformanceService(
        engine_repository=FileSystemInverseMappingEngineRepository(),
        data_repository=FileSystemDatasetRepository(),
        visualizer=visualizer,
    )
