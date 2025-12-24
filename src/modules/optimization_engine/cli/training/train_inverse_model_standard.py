import click

from ...application.factories.estimator import EstimatorFactory
from ...application.factories.metrics import MetricFactory
from ...application.training.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from ...application.training.train_inverse_model.command import (
    TrainInverseModelCommand,
)
from ...application.training.train_inverse_model.handler import (
    TrainInverseModelCommandHandler,
)
from ...domain.modeling.enums.estimator_key import EstimatorKeyEnum
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


@click.command(help="Train inverse model using a single train/test split")
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
    handler = TrainInverseModelCommandHandler(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )

    command = TrainInverseModelCommand(
        dataset_name=dataset_name,
        estimator_params=ESTIMATOR_PARAM_REGISTRY[EstimatorKeyEnum(estimator)](),
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
