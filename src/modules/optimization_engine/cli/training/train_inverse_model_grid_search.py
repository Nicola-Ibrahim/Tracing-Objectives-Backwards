import click

from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)
from ...application.training.train_inverse_model_grid_search.command import (
    TrainInverseModelGridSearchCommand,
)
from ...application.training.train_inverse_model_grid_search.handler import (
    TrainInverseModelGridSearchCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")
INVERSE_ESTIMATOR_REGISTRY: dict[str, type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}
DEFAULT_TUNE_PARAM_NAME = "n_neighbors"
DEFAULT_TUNE_PARAM_RANGE = [5, 10, 20, 40]


def _create_estimator_params(estimation: str) -> EstimatorParams:
    try:
        params_cls = INVERSE_ESTIMATOR_REGISTRY[estimation]
    except KeyError as exc:  # pragma: no cover - enforced by click.Choice
        raise click.BadParameter(
            f"Unsupported estimation '{estimation}'", param_hint="--estimation"
        ) from exc
    return params_cls()


def _default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=name, params={})
        for name in DEFAULT_VALIDATION_METRICS
    ]


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
    "--estimation",
    type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
    default="mdn",
    show_default=True,
    help="Which estimator configuration to use.",
)
@click.option(
    "--dataset-name",
    default="dataset",
    show_default=True,
    help="Dataset identifier to load for training.",
)
def cli(estimation: str, dataset_name: str) -> None:
    handler = _build_inverse_grid_handler()
    estimator_params = _create_estimator_params(estimation)
    command = TrainInverseModelGridSearchCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_default_metric_configs(),
        tune_param_name=DEFAULT_TUNE_PARAM_NAME,
        tune_param_range=DEFAULT_TUNE_PARAM_RANGE,
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
