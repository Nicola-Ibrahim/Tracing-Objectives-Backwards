from pydantic import BaseModel, Field

from ...domain.value_objects.estimator_params import (
    EstimatorParams,
    ValidationMetricConfig,
)


class TrainForwardModelParams(BaseModel):
    """Payload for standard forward (decisions ➝ objectives) training."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the processed dataset to use for training.",
        examples=["dataset"],
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the forward estimator.",
        examples=[{"type": "mdn"}],
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        ...,
        description="Validation metrics to compute during training.",
        examples=[[{"type": "MSE", "params": {}}, {"type": "MAE", "params": {}}]],
    )

    random_state: int = Field(
        ...,
        description="Random seed used across train/test split & estimators.",
        examples=[42],
    )

    learning_curve_steps: int = Field(
        ...,
        description="Number of learning-curve steps for deterministic estimators.",
        examples=[50],
    )

    epochs: int = Field(
        ...,
        description="Epoch count for probabilistic estimators.",
        examples=[100],
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.entities.model_artifact import ModelArtifact
from ...domain.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.interfaces.base_repository import BaseModelArtifactRepository
from ...domain.services.deterministic import DeterministicModelTrainer
from ...domain.services.probabilistic import ProbabilisticModelTrainer
from ..factories.estimator import EstimatorFactory
from ..factories.metrics import MetricFactory


class TrainForwardModelService:
    """Train, evaluate, and persist forward (decision ➝ objective) estimators."""

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseModelArtifactRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        metric_factory: MetricFactory,
    ) -> None:
        self._processed_data_repository = processed_data_repository
        self._model_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._metric_factory = metric_factory

    def execute(self, params: TrainForwardModelParams) -> None:
        dataset: Dataset = self._processed_data_repository.load(
            name=params.dataset_name
        )
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for training."
            )
        processed_data = dataset.processed

        self._logger.log_info(
            "Training forward model with single train/test split (decisions ➝ objectives)."
        )

        X_train = processed_data.decisions_train
        y_train = processed_data.objectives_train
        X_test = processed_data.decisions_test
        y_test = processed_data.objectives_test
        mapping_direction = "forward"

        estimator_params = params.estimator_params
        metric_configs = [
            cfg.model_dump() for cfg in params.estimator_performance_metric_configs
        ]
        random_state = params.random_state
        learning_curve_steps = params.learning_curve_steps

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        if isinstance(estimator, ProbabilisticEstimator):
            fitted_estimator, training_history, metrics = (
                ProbabilisticModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                )
            )
        elif isinstance(estimator, DeterministicEstimator):
            fitted_estimator, training_history, metrics = (
                DeterministicModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    learning_curve_steps=learning_curve_steps,
                    validation_metrics=validation_metrics,
                    random_state=random_state,
                )
            )
        else:  # pragma: no cover - defensive guard for unknown estimator types
            raise TypeError(f"Unsupported estimator type: {type(estimator)!r}")

        self._logger.log_info("Model training (single split) completed.")

        artifact = ModelArtifact.create(
            parameters=params.estimator_params,
            estimator=fitted_estimator,
            metrics=metrics,
            training_history=training_history,
            mapping_direction=mapping_direction,
            dataset_name=params.dataset_name,
        )

        self._model_repository.save(artifact)
