import click

from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from ...application.training.train_inverse_model_grid_search.command import (
    TrainInverseModelGridSearchCommand,
)
from ...application.training.train_inverse_model_grid_search.handler import (
    TrainInverseModelGridSearchCommandHandler,
)
from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...domain.modeling.value_objects.estimator_params import EstimatorParams
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger


@click.command(help="Train inverse model with grid search + cross-validation")
@click.option(
    "--estimator",
    type=click.Choice([k.value for k in ESTIMATOR_PARAM_REGISTRY.keys()]),
    default=EstimatorTypeEnum.MDN.value,
    show_default=True,
    help="Which estimator configuration to use.",
)
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to load for training.",
)
def cli(estimator: str, dataset_name: str) -> None:
    command = TrainInverseModelGridSearchCommand(
        dataset_name=dataset_name,
        estimator_params=ESTIMATOR_PARAM_REGISTRY[EstimatorTypeEnum(estimator)](),
        estimator_performance_metric_configs=default_metric_configs(),
        tune_param_name="n_neighbors",
        tune_param_range=[5, 10, 20, 40],
        random_state=42,
        cv_splits=5,
        learning_curve_steps=50,
        epochs=100,
    )
    handler = TrainInverseModelGridSearchCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationGridCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
