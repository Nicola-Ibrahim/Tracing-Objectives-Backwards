from pydantic import BaseModel, Field

from modules.modeling.domain.value_objects.estimator_params import (
    EstimatorParams,
    ValidationMetricConfig,
)


class TrainInverseModelCrossValidationParams(BaseModel):
    """Payload for k-fold inverse (objectives ➝ decisions) training."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the processed dataset to use for training.",
        examples=["dataset"],
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the inverse estimator.",
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

    cv_splits: int = Field(
        ...,
        ge=2,
        description="Number of cross-validation splits.",
        examples=[5],
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


from modules.shared.domain.interfaces.base_logger import BaseLogger
from modules.dataset.domain.entities.dataset import Dataset
from modules.dataset.domain.interfaces.base_repository import BaseDatasetRepository
from modules.modeling.domain.entities.model_artifact import ModelArtifact
from modules.modeling.domain.interfaces.base_repository import BaseModelArtifactRepository
from modules.modeling.domain.services.cross_validation import CrossValidationTrainer
from ...factories.estimator import EstimatorFactory
from ...factories.metrics import MetricFactory


class TrainInverseModelCrossValidationService:
    """Train, evaluate, and persist inverse estimators using k-fold CV."""

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

    def execute(self, params: TrainInverseModelCrossValidationParams) -> None:
        dataset: Dataset = self._processed_data_repository.load(
            name=params.dataset_name
        )
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for training."
            )
        processed_data = dataset.processed

        self._logger.log_info(
            "Training inverse model with cross-validation (objectives ➝ decisions)."
        )

        X_train = processed_data.objectives_train
        y_train = processed_data.decisions_train
        X_test = processed_data.objectives_test
        y_test = processed_data.decisions_test
        mapping_direction = "inverse"

        estimator_params = params.estimator_params
        metric_configs = [
            cfg.model_dump() for cfg in params.estimator_performance_metric_configs
        ]
        random_state = params.random_state
        cv_splits = params.cv_splits
        learning_curve_steps = params.learning_curve_steps
        epochs = params.epochs

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        fitted_estimator, loss_history, metrics = CrossValidationTrainer().validate(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            validation_metrics=validation_metrics,
            epochs=epochs,
            n_splits=cv_splits,
            random_state=random_state,
            learning_curve_steps=learning_curve_steps,
        )
        self._logger.log_info("Cross-validation workflow completed.")

        artifact = ModelArtifact.create(
            parameters=params.estimator_params,
            estimator=fitted_estimator,
            metrics=metrics,
            training_history=loss_history,
            mapping_direction=mapping_direction,
            dataset_name=params.dataset_name,
            run_metadata={"cv_splits": cv_splits},
        )

        self._model_repository.save(artifact)
