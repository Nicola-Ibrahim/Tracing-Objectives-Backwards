import click

from ...application.visualizing.visualize_model_performance.command import (
    VisualizeModelPerformanceCommand,
)
from ...application.visualizing.visualize_model_performance.handler import (
    VisualizeModelPerformanceCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.modeling.visualizers.performance.vis_2d.visualizer import (
    ModelPerformance2DVisualizer,
)


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
@click.option(
    "--model-number",
    type=int,
    default=None,
    help="Nth most recent model to visualize (1 = latest). Defaults to latest.",
)
def main(
    estimator_name: str,
    dataset_name: str,
    mapping_direction: str,
    model_number: int | None,
) -> None:
    handler = VisualizeModelPerformanceCommandHandler(
        model_artificat_repo=FileSystemModelArtifactRepository(),
        processed_dataset_repo=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
    )
    command = VisualizeModelPerformanceCommand(
        estimator_type=EstimatorTypeEnum(estimator_name),
        processed_file_name=dataset_name,
        data_file_name=dataset_name,
        mapping_direction=mapping_direction,
        model_number=model_number,
    )
    handler.execute(command)


if __name__ == "__main__":
    main()
