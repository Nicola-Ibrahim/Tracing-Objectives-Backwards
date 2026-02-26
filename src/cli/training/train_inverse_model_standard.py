import click

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.modeling.application.registry import (
    ESTIMATOR_PARAM_REGISTRY,
    default_metric_configs,
)
from ...modules.modeling.application.use_cases import (
    TrainInverseModelParams,
    TrainInverseModelService,
)
from ...modules.modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modules.modeling.domain.services.preprocessing_service import (
    PreprocessingService,
)
from ...modules.modeling.infrastructure.factories.estimator import EstimatorFactory
from ...modules.modeling.infrastructure.factories.metrics import MetricFactory
from ...modules.modeling.infrastructure.factories.transformer import TransformerFactory
from ...modules.modeling.infrastructure.repositories.trained_pipeline_repo import (
    FileSystemTrainedPipelineRepository,
)
from ...modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


@click.command(help="Train inverse model using a single train/test split")
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
    params = TrainInverseModelParams(
        dataset_name=dataset_name,
        estimator_params=ESTIMATOR_PARAM_REGISTRY[EstimatorTypeEnum(estimator)](),
        estimator_performance_metric_configs=default_metric_configs(),
        transforms=[
            {"target": "objectives", "type": "min_max", "params": {}},
            {"target": "decisions", "type": "min_max", "params": {}},
        ],
        random_state=42,
        learning_curve_steps=50,
        epochs=100,
    )

    service = TrainInverseModelService(
        processed_data_repository=FileSystemDatasetRepository(),
        model_repository=FileSystemTrainedPipelineRepository(),
        logger=CMDLogger(name="InterpolationCMDLogger"),
        estimator_factory=EstimatorFactory(),
        metric_factory=MetricFactory(),
        transformer_factory=TransformerFactory(),
        preprocessing_service=PreprocessingService(),
    )

    service.execute(params)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
