import click

from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from ...application.training.train_forward_model.command import (
    TrainForwardModelCommand,
)
from ...application.training.train_forward_model.handler import (
    TrainForwardModelCommandHandler,
)
from ...domain.modeling.enums.estimator_key import EstimatorKeyEnum
from ...infrastructure.datasets.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...infrastructure.modeling.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from ...infrastructure.shared.loggers.cmd_logger import CMDLogger

FORWARD_ESTIMATOR_KEYS: tuple[EstimatorKeyEnum, ...] = tuple(
    ESTIMATOR_PARAM_REGISTRY.keys()
)


@click.group(help="Train a forward model (single train/test split workflow)")
def cli() -> None:
    return None


@cli.command(name="standard", help="Single train/test split training")
@click.option(
    "--estimator",
    type=click.Choice(sorted([k.value for k in FORWARD_ESTIMATOR_KEYS])),
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
def command_standard(estimator: str, dataset_name: str) -> None:
    handler = TrainForwardModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ForwardCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )
    estimator_key = EstimatorKeyEnum(estimator)
    estimator_params = ESTIMATOR_PARAM_REGISTRY[estimator_key]()
    command = TrainForwardModelCommand(
        dataset_name=dataset_name,
        estimator_params=estimator_params,
        estimator_performance_metric_configs=default_metric_configs(),
        random_state=42,
        learning_curve_steps=50,
        epochs=100,
    )
    handler.execute(command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
