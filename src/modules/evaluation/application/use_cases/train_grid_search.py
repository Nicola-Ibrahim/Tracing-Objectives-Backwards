from typing import Any

from pydantic import BaseModel, Field

from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modeling.application.factories.estimator import EstimatorFactory
from ....modeling.application.factories.metrics import MetricFactory
from ....modeling.application.factories.normalizer import NormalizerFactory
from ....modeling.application.registry import EstimatorParams
from ....modeling.domain.entities.trained_pipeline import TrainedPipeline
from ....modeling.domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ....modeling.domain.interfaces.base_transform import TransformTarget
from ....modeling.domain.services.cross_validation import CrossValidationTrainer
from ....modeling.domain.services.preprocessing_service import PreprocessingService
from ....modeling.domain.value_objects.estimator_params import ValidationMetricConfig
from ....modeling.domain.value_objects.split_step import SplitConfig
from ....shared.domain.interfaces.base_logger import BaseLogger


class TrainInverseModelGridSearchParams(BaseModel):
    """Payload for grid-search training of inverse (objectives ➝ decisions) estimators."""

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
        description="Number of cross-validation splits to use during grid search.",
    )

    tune_param_name: str = Field(
        ...,
        description="Hyperparameter name to tune.",
    )

    tune_param_range: list[Any] = Field(
        ...,
        description="Candidate values for the tuned hyperparameter.",
        min_items=1,
    )

    learning_curve_steps: int = Field(
        ...,
        description="Number of learning-curve steps for deterministic estimators during grid search.",
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


class TrainInverseModelGridSearchService:
    """Train, tune, and persist inverse estimators using grid search."""

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

    def execute(self, params: TrainInverseModelGridSearchParams) -> None:
        dataset: Dataset = self._data_repository.load(name=params.dataset_name)

        self._logger.log_info(
            "Training inverse model with grid search (objectives ➝ decisions)."
        )

        X_raw = dataset.objectives
        y_raw = dataset.decisions
        mapping_direction = "inverse"

        params.split_config.random_state = params.random_state
        split_step, X_train, X_test, y_train, y_test = (
            self._preprocessing_service.split(X_raw, y_raw, params.split_config)
        )

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
        tune_param_name = params.tune_param_name
        tune_param_range = params.tune_param_range
        learning_curve_steps = params.learning_curve_steps
        epochs = params.epochs

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        parameters_for_search = params.estimator_params.model_dump()

        (
            fitted_estimator,
            training_log,
            evaluation_result,
            search_summary,
        ) = CrossValidationTrainer().search(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            param_name=tune_param_name,
            param_range=tune_param_range,
            validation_metrics=validation_metrics,
            parameters=parameters_for_search,
            random_state=random_state,
            cv=cv_splits,
            epochs=epochs,
            learning_curve_steps=learning_curve_steps,
        )
        self._logger.log_info("Grid search workflow completed.")

        from ....modeling.domain.value_objects.estimator_step import EstimatorStep

        estimator_step = EstimatorStep(
            config=params.estimator_params,
            fitted=fitted_estimator,
            training_log=training_log,
        )

        pipeline = TrainedPipeline(
            dataset_name=params.dataset_name,
            mapping_direction=mapping_direction,
            split=split_step,
            transforms=transforms,
            model=estimator_step,
            evaluation=evaluation_result,
            run_metadata={
                "cv_splits": cv_splits,
                "grid_searched_param": tune_param_name,
                "grid_search_summary": search_summary,
            },
        )

        self._pipeline_repository.save(pipeline)
