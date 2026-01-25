import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.evaluation.application.use_cases.visualize_model_performance import (
    VisualizeModelPerformanceCommand,
    VisualizeModelPerformanceCommandHandler,
)
from ...modules.evaluation.infrastructure.visualization.model_performance_2d.visualizer import (
    ModelPerformance2DVisualizer,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)


@click.command(help="Visualize a trained model's performance diagnostics")
@click.option(
    "--estimator",
    "estimator_name",
    required=True,
    help="Estimator type to visualize (matches stored artifact type)",
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
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to visualize.",
)
def main(
    estimator_name: str,
    mapping_direction: str,
    model_number: int | None,
    dataset_name: str,
) -> None:
    handler = VisualizeModelPerformanceCommandHandler(
        model_artificat_repo=FileSystemModelArtifactRepository(),
        processed_dataset_repo=FileSystemDatasetRepository(),
        visualizer=ModelPerformance2DVisualizer(),
    )
    command = VisualizeModelPerformanceCommand(
        dataset_name=dataset_name,
        data_file_name=dataset_name,
        processed_file_name=dataset_name,
        estimator_type=EstimatorTypeEnum(estimator_name),
        mapping_direction=mapping_direction,
        model_number=model_number,
    )
    handler.execute(command)


if __name__ == "__main__":
    main()
