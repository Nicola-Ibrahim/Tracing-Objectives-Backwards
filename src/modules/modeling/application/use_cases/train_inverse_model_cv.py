from typing import Any

from pydantic import BaseModel, Field

from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.entities.trained_pipeline import TrainedPipeline
from ...domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ...domain.interfaces.base_transform import TransformTarget
from ...domain.services.cross_validation import CrossValidationTrainer
from ...domain.services.preprocessing_service import PreprocessingService
from ...domain.value_objects.estimator_params import ValidationMetricConfig
from ...domain.value_objects.split_step import SplitConfig
from ..factories.estimator import EstimatorFactory
from ..factories.metrics import MetricFactory
from ..factories.normalizer import NormalizerFactory
from ..registry import EstimatorParams


class TrainInverseModelCrossValidationParams(BaseModel):
    """Payload for k-fold inverse (objectives ➝ decisions) training."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the dataset to use for training.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure the inverse estimator.",
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        ...,
        description="Validation metrics to compute during training.",
    )

    random_state: int = Field(
        ...,
        description="Random seed used across train/test split & estimators.",
    )

    cv_splits: int = Field(
        ...,
        ge=2,
        description="Number of cross-validation splits.",
    )

    learning_curve_steps: int = Field(
        ...,
        description="Number of learning-curve steps for deterministic estimators.",
    )

    epochs: int = Field(
        ...,
        description="Epoch count for probabilistic estimators.",
    )

    split_config: SplitConfig = Field(
        default_factory=SplitConfig,
        description="Configuration for train/test splitting.",
    )

    decisions_normalizer: dict[str, Any] = Field(
        default_factory=lambda: {"type": "min_max", "params": {}},
        description="Normalizer configuration for decisions.",
    )

    objectives_normalizer: dict[str, Any] = Field(
        default_factory=lambda: {"type": "min_max", "params": {}},
        description="Normalizer configuration for objectives.",
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class TrainInverseModelCrossValidationService:
    """Train, evaluate, and persist inverse estimators using k-fold CV."""

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseTrainedPipelineRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        metric_factory: MetricFactory,
        normalizer_factory: NormalizerFactory,
        preprocessing_service: PreprocessingService,
    ) -> None:
        self._data_repository = processed_data_repository
        self._pipeline_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._metric_factory = metric_factory
        self._normalizer_factory = normalizer_factory
        self._preprocessing_service = preprocessing_service

    def execute(self, params: TrainInverseModelCrossValidationParams) -> None:
        dataset: Dataset = self._data_repository.load(name=params.dataset_name)

        self._logger.log_info(
            "Training inverse model with cross-validation (objectives ➝ decisions)."
        )

        # Inverse mapping: X=objectives, y=decisions
        X_raw = dataset.objectives
        y_raw = dataset.decisions

        mapping_direction = "inverse"

        # Split
        params.split_config.random_state = params.random_state
        split_step, X_train, X_test, y_train, y_test = (
            self._preprocessing_service.split(X_raw, y_raw, params.split_config)
        )

        # Transforms
        obj_transform = self._normalizer_factory.create(
            params.objectives_normalizer, TransformTarget.OBJECTIVES
        )
        dec_transform = self._normalizer_factory.create(
            params.decisions_normalizer, TransformTarget.DECISIONS
        )
        transforms = [obj_transform, dec_transform]

        X_train, X_test, y_train, y_test = self._preprocessing_service.apply_transforms(
            X_train, X_test, y_train, y_test, transforms
        )

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

        fitted_estimator, training_log, evaluation_result = (
            CrossValidationTrainer().validate(
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
        )
        self._logger.log_info("Cross-validation workflow completed.")

        from ...domain.value_objects.estimator_step import EstimatorStep

        estimator_step = EstimatorStep(
            config=params.estimator_params,
            fitted=fitted_estimator,
            training_log=training_log,
        )

        cv_run_metadata = {"cv_splits": cv_splits}

        pipeline = TrainedPipeline(
            dataset_name=params.dataset_name,
            mapping_direction=mapping_direction,
            split=split_step,
            transforms=transforms,
            model=estimator_step,
            evaluation=evaluation_result,
            run_metadata=cv_run_metadata,
        )

        self._pipeline_repository.save(pipeline)
