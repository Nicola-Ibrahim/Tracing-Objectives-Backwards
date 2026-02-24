import click

from modules.modeling.application.factories.estimator import EstimatorFactory
from modules.modeling.application.factories.metrics import MetricFactory
from modules.modeling.application.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from modules.modeling.application.train_forward_model.service import (
    TrainForwardModelParams,
)
from modules.modeling.application.train_forward_model.handler import (
    TrainForwardModelService,
)
from modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from modules.modeling.infrastructure.repositories.model_artifact_repo import (
    FileSystemModelArtifactRepository,
)
from modules.shared.infrastructure.loggers.cmd_logger import CMDLogger

FORWARD_ESTIMATOR_KEYS: tuple[EstimatorTypeEnum, ...] = tuple()


@click.group(help="Train a forward model (single train/test split workflow)")
def cli() -> None:
    return None


@cli.service(name="standard", help="Single train/test split training")
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
def command_standard(estimator: str, dataset_name: str) -> None:
    service = TrainForwardModelService(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemModelArtifactRepository(),
        logger=CMDLogger(name="ForwardCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
    )
    estimator_params = ESTIMATOR_PARAM_REGISTRY[EstimatorTypeEnum(estimator)]()
    params = TrainForwardModelParams(
        dataset_name=dataset_name,
        estimator_params=estimator_params,
        estimator_performance_metric_configs=default_metric_configs(),
        random_state=42,
        learning_curve_steps=50,
        epochs=100,
    )
    service.execute(params)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
