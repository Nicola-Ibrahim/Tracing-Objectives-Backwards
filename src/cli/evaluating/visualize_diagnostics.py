import click

from ...modules.evaluation.application.use_cases.visualize_diagnostics import (
    InverseEstimatorCandidate,
    VisualizeInverseEstimatorDiagnosticCommand,
    VisualizeInverseEstimatorDiagnosticCommandHandler,
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

    command = VisualizeInverseEstimatorDiagnosticCommand(
        dataset_name="cocoex_f5",
        requests=[
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=1, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=2, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=3, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=4, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=5, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=6, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=7, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=8, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=9, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=10, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.MDN, version=11, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.CVAE, version=1, run_number=1
            ),
            InverseEstimatorCandidate(
                type=EstimatorTypeEnum.INN, version=1, run_number=1
            ),
        ],
    )

    handler = VisualizeInverseEstimatorDiagnosticCommandHandler(
        diagnostic_repository=FileSystemDiagnosticRepository(),
        visualizer=InverseModelsComparisonVisualizer(),
        logger=logger,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
