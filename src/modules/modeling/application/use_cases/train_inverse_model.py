from typing import Any

from pydantic import BaseModel, Field

from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.entities.trained_pipeline import TrainedPipeline
from ...domain.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.interfaces.base_repository import (
    BaseTrainedPipelineRepository,
)
from ...domain.services.deterministic import DeterministicModelTrainer
from ...domain.services.preprocessing_service import PreprocessingService
from ...domain.services.probabilistic import ProbabilisticModelTrainer
from ...domain.value_objects.estimator import Estimator
from ...domain.value_objects.estimator_params import (
    ValidationMetricConfig,
)
from ...domain.value_objects.split_step import SplitConfig
from ...infrastructure.factories.estimator import EstimatorFactory
from ...infrastructure.factories.metrics import MetricFactory
from ...infrastructure.factories.transformer import TransformerFactory
from ..registry import EstimatorParams


class TrainInverseModelParams(BaseModel):
    """Command payload for single-split inverse (objectives ➝ decisions) training."""

    dataset_name: str = Field(
        ...,
        description="Identifier of the dataset to use for training.",
    )

    estimator_params: EstimatorParams = Field(
        ...,
        description="Parameters used to initialize/configure this specific inverse estimator.",
    )

    estimator_performance_metric_configs: list[ValidationMetricConfig] = Field(
        ...,
        description="Configurations for the validation metrics.",
    )

    random_state: int = Field(
        ...,
        description="Random seed used across train/test split & estimators.",
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

    transforms: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of ordered transform configuration dictionaries.",
    )

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class TrainInverseModelService:
    """Train, evaluate, and persist inverse (objectives ➝ decisions) estimators."""

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseTrainedPipelineRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        metric_factory: MetricFactory,
        transformer_factory: TransformerFactory,
        preprocessing_service: PreprocessingService,
    ) -> None:
        self._data_repository = processed_data_repository
        self._pipeline_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._metric_factory = metric_factory
        self._transformer_factory = transformer_factory
        self._preprocessing_service = preprocessing_service

    def execute(self, params: TrainInverseModelParams) -> None:
        dataset: Dataset = self._data_repository.load(name=params.dataset_name)

        self._logger.log_info("Training inverse model (objectives ➝ decisions).")

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
        decision_transforms = []
        objective_transforms = []
        for t_cfg in params.transforms:
            target = t_cfg.get("target")
            transform = self._transformer_factory.create(t_cfg)
            # Set target attribute for persistence/filtering
            setattr(transform, "target", target)

            if target in ("decisions", "both"):
                decision_transforms.append(transform)
            if target in ("objectives", "both"):
                objective_transforms.append(transform)

        X_train, X_test = self._preprocessing_service.apply_transforms(
            X_train, X_test, decision_transforms
        )
        y_train, y_test = self._preprocessing_service.apply_transforms(
            y_train, y_test, objective_transforms
        )

        estimator_params = params.estimator_params
        metric_configs = [
            cfg.model_dump() for cfg in params.estimator_performance_metric_configs
        ]
        random_state = params.random_state
        learning_curve_steps = params.learning_curve_steps
        epochs = params.epochs

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        self._logger.log_info("Starting single train/test split workflow.")
        if isinstance(estimator, ProbabilisticEstimator):
            fitted_estimator, training_log, evaluation_result = (
                ProbabilisticModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    epochs=epochs,
                )
            )

        elif isinstance(estimator, DeterministicEstimator):
            fitted_estimator, training_log, evaluation_result = (
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
        else:
            raise TypeError(f"Unsupported estimator type: {type(estimator)!r}")

        self._logger.log_info("Model training (single split) completed.")

        estimator = Estimator(
            config=params.estimator_params,
            fitted=fitted_estimator,
            training_log=training_log,
        )

        pipeline = TrainedPipeline(
            dataset_name=params.dataset_name,
            mapping_direction=mapping_direction,
            split=split_step,
            transforms=decision_transforms + objective_transforms,
            estimator=estimator,
            evaluation=evaluation_result,
        )
        self._pipeline_repository.save(pipeline)
