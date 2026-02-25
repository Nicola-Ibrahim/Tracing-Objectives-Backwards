import click

from src.modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from src.modules.modeling.application.factories.estimator import EstimatorFactory
from src.modules.modeling.application.factories.metrics import MetricFactory
from src.modules.modeling.application.factories.normalizer import NormalizerFactory
from src.modules.modeling.application.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from src.modules.modeling.application.use_cases import (
    TrainForwardModelParams,
    TrainForwardModelService,
)
from src.modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from src.modules.modeling.domain.services.preprocessing_service import (
    PreprocessingService,
)
from src.modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from src.modules.shared.infrastructure.loggers.cmd_logger import CMDLogger

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
        model_repository=FileSystemTrainedPipelineRepository(),
        logger=CMDLogger(name="ForwardCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
        normalizer_factory=NormalizerFactory(),
        preprocessing_service=PreprocessingService(),
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
