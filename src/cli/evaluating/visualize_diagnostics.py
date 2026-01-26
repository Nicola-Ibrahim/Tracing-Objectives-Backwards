import click

from ...modules.evaluation.application.use_cases.visualize_diagnostics import (
    EstimatorDiagnosticRequest,
    VisualizeDiagnosticsCommand,
    VisualizeDiagnosticsCommandHandler,
)
from ...modules.evaluation.infrastructure.repositories.diagnostic_repository import (
    FileSystemDiagnosticRepository,
)
from ...modules.evaluation.infrastructure.visualization.inverse_comparison.visualizer import (
    InverseComparisonVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Visualize historical diagnostic evaluation runs")
@click.option(
    "--dataset-name",
    default="cocoex_f5",
    help="The dataset context results belong to.",
)
@click.option(
    "--estimator",
    "-e",
    multiple=True,
    nargs=3,
    type=(str, int, int),
    help="Historical run to load: (type, version, run_number). Use 0 for latest run.",
)
def cli(dataset_name, estimator):
    """
    CLI to render plots from previously saved evaluation JSONs.
    Example: python -m src.cli.evaluating.visualize_diagnostics --dataset-name cocoex_f5 -e mdn 1 1 -e cvae 1 2
    """
    logger = CMDLogger(name="VisualizeDiagnosticsLogger")

    # Map raw tuples to requests
    requests = []
    for type_str, version, run_num in estimator:
        try:
            req = EstimatorDiagnosticRequest(
                type=EstimatorTypeEnum(type_str),
                version=version,
                run_number=run_num if run_num > 0 else None,
            )
            requests.append(req)
        except ValueError as e:
            logger.log_error(f"Invalid estimator type '{type_str}': {e}")
            return

    if not requests:
        logger.log_warning("No diagnostic requests provided. Use -e or --estimator.")
        return

    command = VisualizeDiagnosticsCommand(
        dataset_name=dataset_name,
        requests=requests,
    )

    handler = VisualizeDiagnosticsCommandHandler(
        diagnostic_repository=FileSystemDiagnosticRepository(),
        visualizer=InverseComparisonVisualizer(),
        logger=logger,
    )

    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
