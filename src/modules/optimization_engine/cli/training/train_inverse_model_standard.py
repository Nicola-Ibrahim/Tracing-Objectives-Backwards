import click

from ...application.dtos import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParams,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)
from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.train_inverse_model.command import (
    TrainInverseModelCommand,
)
from ...application.training.train_inverse_model.handler import (
    TrainInverseModelCommandHandler,
)
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)

DEFAULT_VALIDATION_METRICS: tuple[str, ...] = ("MSE", "MAE", "R2")
INVERSE_ESTIMATOR_REGISTRY: dict[str, type[EstimatorParams]] = {
    "cvae": CVAEEstimatorParams,
    "gaussian_process": GaussianProcessEstimatorParams,
    "mdn": MDNEstimatorParams,
    "neural_network": NeuralNetworkEstimatorParams,
    "rbf": RBFEstimatorParams,
    "coco": COCOEstimatorParams,
}


def _default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=name, params={})
        for name in DEFAULT_VALIDATION_METRICS
    ]


@click.command(help="Train inverse model using a single train/test split")
@click.option(
    "--estimation",
    type=click.Choice(sorted(INVERSE_ESTIMATOR_REGISTRY.keys())),
    default="mdn",
    show_default=True,
    help="Which estimator configuration to use.",
)
def cli(estimation: str) -> None:
    handler = TrainInverseModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )

    command = TrainInverseModelCommand(
        estimator_params=INVERSE_ESTIMATOR_REGISTRY[estimation](),
        estimator_performance_metric_configs=_default_metric_configs(),
    )
    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
