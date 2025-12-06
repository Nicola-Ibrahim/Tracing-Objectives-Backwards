import click

from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    CVAEMDNEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.train_inverse_model_cv.command import (
    TrainInverseModelCrossValidationCommand,
)
from ...application.training.train_inverse_model_cv.handler import (
    TrainInverseModelCrossValidationCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")
INVERSE_ESTIMATOR_REGISTRY: dict[str, type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "cvae_mdn": CVAEMDNEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}


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


def _build_inverse_cv_handler() -> TrainInverseModelCrossValidationCommandHandler:
    return TrainInverseModelCrossValidationCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCVCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )


@click.command(help="Train inverse model with k-fold cross-validation")
@click.option(
    "--estimation",
    type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
    default="mdn",
    show_default=True,
    help="Which estimator configuration to use.",
)
def cli(estimation: str) -> None:
    handler = _build_inverse_cv_handler()
    estimator_params = _create_estimator_params(estimation)
    command = TrainInverseModelCrossValidationCommand(
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_default_metric_configs(),
    )
    handler.execute(command)


def main() -> None:  # pragma: no cover - CLI entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
