import click

from ...domain.modeling.value_objects.estimator_params import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParamsBase,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.train_forward_model.command import (
    TrainForwardModelCommand,
)
from ...application.training.train_forward_model.handler import (
    TrainForwardModelCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")
FORWARD_ESTIMATOR_REGISTRY: dict[str, type[EstimatorParamsBase]] = {
    "coco": COCOEstimatorParams,
    "cvae": CVAEEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
}


def _default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=name, params={})
        for name in DEFAULT_VALIDATION_METRICS
    ]


@click.group(help="Train a forward model (single train/test split workflow)")
def cli() -> None:
    return None


@cli.command(name="standard", help="Single train/test split training")
@click.option(
    "--estimation",
    type=click.Choice(sorted(FORWARD_ESTIMATOR_REGISTRY.keys())),
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
def command_standard(estimation: str, dataset_name: str) -> None:
    handler = TrainForwardModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ForwardCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )
    estimator_params = FORWARD_ESTIMATOR_REGISTRY[estimation]()
    command = TrainForwardModelCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params,
        estimator_performance_metric_configs=_default_metric_configs(),
    )
    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
