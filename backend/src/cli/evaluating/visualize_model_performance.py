import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.check_engine_performance import (
    CheckModelPerformanceParams,
    CheckModelPerformanceService,
)
from ...modules.evaluation.application.diagnose_engines import EngineCandidate
from ...modules.evaluation.infrastructure.visualization.model_performance_2d.visualizer import (
    ModelPerformance2DVisualizer,
)
from ...modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)


@click.command(help="Visualize a trained engine's performance diagnostics")
def main() -> None:
    service = CheckModelPerformanceService(
        engine_repository=FileSystemInverseMappingEngineRepository(),
        data_repository=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
    )
    params = CheckModelPerformanceParams(
        dataset_name="cocoex_f5",
        engine=EngineCandidate(solver_type="GBPI", version=1),
        n_samples=50,
    )
    service.execute(params)


if __name__ == "__main__":
    main()
