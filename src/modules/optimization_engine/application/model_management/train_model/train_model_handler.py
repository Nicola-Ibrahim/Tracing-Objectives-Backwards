from typing import Any

import numpy as np

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_management.interfaces.base_estimator import BaseEstimator
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_management.interfaces.base_normalizer import BaseNormalizer
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ...factories.estimator import EstimatorFactory
from ...factories.mertics import MetricFactory
from ...factories.normalizer import NormalizerFactory
from ...services.model_training import TrainerService
from .train_model_command import TrainModelCommand


class TrainModelCommandHandler:
    """
    Orchestrates training, validation, logging, visualization, and persistence
    of inverse decision mapping models.

    Responsibilities:
        - Fetch training data
        - Normalize and split data
        - Train model(s) with single split or CV
        - Evaluate with configured metrics
        - Save artifacts & visualize results
    """

    def __init__(
        self,
        data_repository: BaseParetoDataRepository,
        model_repository: BaseInterpolationModelRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        normalizer_factory: NormalizerFactory,
        metric_factory: MetricFactory,
        visualizer: BaseDataVisualizer | None = None,
    ) -> None:
        self._data_repository = data_repository
        self._estimator_factory = estimator_factory
        self._logger = logger
        self._model_repository = model_repository
        self._normalizer_factory = normalizer_factory
        self._metric_factory = metric_factory
        self._visualizer = visualizer
        self._trainer_service = TrainerService()

    # --------------------- PUBLIC ENTRY ---------------------

    def execute(self, command: TrainModelCommand) -> None:
        """
        Executes the training workflow for a given command.
        Unpacks command attributes to pass only necessary data to sub-methods.
        """
        raw_data: DataModel = self._data_repository.load(filename="pareto_data")

        # Unpack command attributes once at the highest level
        estimator_params = command.estimator_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.model_performance_metric_configs
        ]
        normalizer_config = command.normalizer_config.model_dump()
        test_size = command.test_size
        random_state = command.random_state
        cv_splits = command.cv_splits

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        # 1) Build normalizers
        objectives_normalizer = self._normalizer_factory.create(
            config=normalizer_config
        )
        decisions_normalizer = self._normalizer_factory.create(config=normalizer_config)

        # 2) Train and evaluate model
        parameters = {**estimator.to_dict(), "type": estimator.type}
        if cv_splits:
            self._logger.log_info("Starting cross-validation workflow.")
            artifact = self._trainer_service.cross_validate(
                estimator=estimator,
                X=raw_data.historical_objectives,
                y=raw_data.historical_solutions,
                X_normalizer=decisions_normalizer,
                y_normalizer=objectives_normalizer,
                validation_metrics=validation_metrics,
                parameters=parameters,
                n_splits=cv_splits,
                random_state=random_state,
                verbose=False,
            )
            self._logger.log_info("Cross-validation workflow completed.")

        else:
            self._logger.log_info("Starting single train/test split workflow.")
            artifact = self._trainer_service.train_and_evaluate(
                estimator=estimator,
                X=raw_data.historical_objectives,
                y=raw_data.historical_solutions,
                metrics=validation_metrics,
                X_normalizer=decisions_normalizer,
                y_normalizer=objectives_normalizer,
                parameters=parameters,
                test_size=test_size,
                random_state=random_state,
                learning_curve_steps=50,
            )

            self._logger.log_info("Model training (single split) completed.")

        # 3) Persist artifact
        self._model_repository.save(artifact)

    def _visualize(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        estimator: BaseEstimator,
        decisions_normalizer: BaseNormalizer,
    ) -> None:
        """Runs visualization if visualizer is provided."""
        if not self._visualizer:
            return

        y_pred_norm = estimator.predict(X_test)
        if isinstance(y_pred_norm, np.ndarray) and y_pred_norm.ndim >= 3:
            y_pred_norm = y_pred_norm.mean(axis=0)
        elif isinstance(y_pred_norm, np.ndarray) and y_pred_norm.ndim == 1:
            y_pred_norm = y_pred_norm.reshape(-1, 1)

        y_pred = decisions_normalizer.inverse_transform(y_pred_norm)

        self._visualizer.plot(
            objectives_train=X_train,
            objectives_val=X_test,
            decisions_train=y_train,
            decisions_val=y_test,
            decisions_pred_val=y_pred,
        )
