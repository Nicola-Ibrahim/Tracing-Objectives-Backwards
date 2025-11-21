import click

from ...application.visualization.visualize_model_performance.command import (
    VisualizeModelPerformanceCommand,
)
from ...application.visualization.visualize_model_performance.handler import (
    VisualizeModelPerformanceCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.modeling.visualizers.performance import (
    ModelPerformance2DVisualizer,
)


def _build_handler() -> VisualizeModelPerformanceCommandHandler:
    return VisualizeModelPerformanceCommandHandler(
        model_artificat_repo=FileSystemModelArtifactRepository(),
        processed_dataset_repo=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
    )


def _coerce_estimator(value: str) -> EstimatorTypeEnum:
    try:
        return EstimatorTypeEnum(value)
    except ValueError as exc:
        valid = ", ".join(e.value for e in EstimatorTypeEnum)
        raise click.BadParameter(
            f"Unknown estimator '{value}'. Choose from: {valid}"
        ) from exc


@click.command(help="Visualize a trained model's performance diagnostics")
@click.option(
    "--estimator",
    "estimator_name",
    required=True,
    help="Estimator type to visualize (matches stored artifact type)",
)
@click.option(
    "--dataset",
    "dataset_name",
    default="dataset",
    show_default=True,
    help="Processed dataset identifier to load",
)
@click.option(
    "--mapping-direction",
    type=click.Choice(["inverse", "forward"]),
    default="inverse",
    show_default=True,
    help="Whether to visualize the inverse (objectives->decisions) or forward (decisions->objectives) model.",
)
def main(estimator_name: str, dataset_name: str, mapping_direction: str) -> None:
    handler = _build_handler()
    command = VisualizeModelPerformanceCommand(
        estimator_type=_coerce_estimator(estimator_name),
        processed_file_name=dataset_name,
        data_file_name=dataset_name,
        mapping_direction=mapping_direction,
    )
    handler.execute(command)


if __name__ == "__main__":
    main()
