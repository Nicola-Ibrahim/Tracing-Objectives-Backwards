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
from ...domain.modeling.enums.estimator_key import EstimatorKeyEnum
from ...domain.modeling.value_objects.estimator_params import EstimatorParams
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger

INVERSE_ESTIMATOR_KEYS: tuple[EstimatorKeyEnum, ...] = tuple(
    ESTIMATOR_PARAM_REGISTRY.keys()
)
DEFAULT_TUNE_PARAM_NAME = "n_neighbors"
DEFAULT_TUNE_PARAM_RANGE = [5, 10, 20, 40]


def _create_estimator_params(estimator: str) -> EstimatorParams:
    try:
        params_cls = ESTIMATOR_PARAM_REGISTRY[EstimatorKeyEnum(estimator)]
    except KeyError as exc:  # pragma: no cover - enforced by click.Choice
        raise click.BadParameter(
            f"Unsupported estimator '{estimator}'", param_hint="--estimator"
        ) from exc
    return params_cls()


def _build_inverse_grid_handler() -> TrainInverseModelGridSearchCommandHandler:
    return TrainInverseModelGridSearchCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationGridCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


@click.command(help="Train inverse model with grid search + cross-validation")
@click.option(
    "--estimator",
    type=click.Choice(sorted([k.value for k in INVERSE_ESTIMATOR_KEYS])),
    default=EstimatorKeyEnum.MDN.value,
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
    handler = _build_inverse_grid_handler()
    estimator_params = _create_estimator_params(estimator)
    command = TrainInverseModelGridSearchCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params,
        estimator_performance_metric_configs=default_metric_configs(),
        tune_param_name=DEFAULT_TUNE_PARAM_NAME,
        tune_param_range=DEFAULT_TUNE_PARAM_RANGE,
        random_state=42,
        cv_splits=5,
        learning_curve_steps=50,
        epochs=100,
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
