import click

from ...modules.evaluation.application.use_cases.visualize_diagnostics import (
    InverseEstimatorCandidate,
    VisualizeInverseEstimatorDiagnosticParams,
    VisualizeInverseEstimatorDiagnosticService,
)
from ...modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ...modules.evaluation.infrastructure.visualization.inverse_models_comparison.visualizer import (
    InverseModelsComparisonVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Visualize historical diagnostic evaluation runs")
def cli():
    """
    CLI to render plots from previously saved evaluation JSONs.
    Configuration is defined explicitly in code.
    """
    logger = CMDLogger(name="VisualizeDiagnosticsLogger")

    params = VisualizeInverseEstimatorDiagnosticParams(
        dataset_name="cocoex_f5",
        inverse_estimator_candidates=[
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=8),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=10),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=12),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.MDN, version=13),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=1),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.CVAE, version=2),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=1),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=3),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=4),
            InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=5),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=6),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=7),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=8),
            # InverseEstimatorCandidate(type=EstimatorTypeEnum.INN, version=9),
        ],
    )

    service = VisualizeInverseEstimatorDiagnosticService(
        diagnostic_repository=FileSystemDiagnosticRepository(),
        visualizer=InverseModelsComparisonVisualizer(),
        logger=logger,
    )

    service.execute(params)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
