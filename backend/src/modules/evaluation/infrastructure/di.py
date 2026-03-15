from dependency_injector import containers, providers

from ..application.check_engine_performance import CheckModelPerformanceService
from ..application.compare_candidates import CompareInverseModelCandidatesService
from ..application.diagnose_engines import RunDiagnosticsService
from ..application.visualize_diagnostics import (
    VisualizeInverseEstimatorDiagnosticService,
)
from .repositories.diagnostic_repository import FileSystemDiagnosticRepository
from .visualization.inverse_models_comparison.visualizer import (
    InverseModelsComparisonVisualizer,
)


class EvaluationContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the Evaluation bounded context.
    Using dependency-injector library.
    """

    engine_repository = providers.Dependency()
    data_repository = providers.Dependency()
    logger = providers.Dependency()

    diagnostic_repository = providers.Singleton(FileSystemDiagnosticRepository)

    # Default visualizer for general comparison
    comparison_visualizer = providers.Singleton(InverseModelsComparisonVisualizer)

    run_diagnostics_service = providers.Factory(
        RunDiagnosticsService,
        engine_repository=engine_repository,
        data_repository=data_repository,
        diagnostic_repository=diagnostic_repository,
        logger=logger,
    )

    compare_candidates_service = providers.Factory(
        CompareInverseModelCandidatesService,
        engine_repository=engine_repository,
        data_repository=data_repository,
        logger=logger,
        visualizer=comparison_visualizer,
    )

    performance_service = providers.Factory(
        CheckModelPerformanceService,
        engine_repository=engine_repository,
        data_repository=data_repository,
        visualizer=comparison_visualizer,
    )

    visualize_diagnostics_service = providers.Factory(
        VisualizeInverseEstimatorDiagnosticService,
        diagnostic_repository=diagnostic_repository,
        logger=logger,
        visualizer=comparison_visualizer,
    )
