from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ....domain.model_management.entities.model_artifact import (
    ModelArtifact,
)
from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_management.interfaces.base_normalizer import BaseNormalizer
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ...factories.inverse_decision_mapper import (
    InverseDecisionMapperFactory,
)
from ...factories.mertics import MetricFactory
from ...factories.normalizer import NormalizerFactory
from .train_model_command import TrainModelCommand


class TrainModelCommandHandler:
    """
    Handler for the TrainModelCommand.
    Orchestrates the interpolator training, validation, logging, and persistence processes.
    Dependencies are injected via the constructor.
    """

    def __init__(
        self,
        pareto_data_repo: BaseParetoDataRepository,
        inverse_decision_factory: InverseDecisionMapperFactory,
        logger: BaseLogger,
        trained_model_repository: BaseInterpolationModelRepository,
        normalizer_factory: NormalizerFactory,
        metric_factory: MetricFactory,
        visualizer: BaseDataVisualizer | None = None,
    ):
        self._pareto_data_repo = pareto_data_repo
        self._inverse_decision_factory = inverse_decision_factory
        self._logger = logger
        self._trained_model_repository = trained_model_repository
        self._normalizer_factory = normalizer_factory
        self._metric_factory = metric_factory
        self._visualizer = visualizer

    def execute(self, command: TrainModelCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.
        """
        self._logger.log_info("Starting single train/test split interpolator training.")

        # Step 1: Initialize all necessary components based on the command
        (
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
        ) = self._initialize_components(
            command.inverse_decision_mapper_params.model_dump(),
            command.objectives_normalizer_config.model_dump(),
            command.decisions_normalizer_config.model_dump(),
        )

        # Step 2: Load and prepare the data
        (
            objectives_train_norm,
            objectives_val_norm,
            decisions_train,
            decisions_val,
        ) = self._prepare_data(
            command.test_size,
            command.random_state,
            objectives_normalizer,
            decisions_normalizer,
        )

        # Step 3: Train the model
        self._train_model(
            inverse_decision_mapper, objectives_train_norm, decisions_train
        )

        # Step 4: Validate the model and calculate metrics
        metrics = self._validate_model(
            inverse_decision_mapper,
            decisions_normalizer,
            objectives_val_norm,
            decisions_val,
            command.model_performance_metric_configs,
        )

        # Step 5: Save the trained model and its metadata (version assigned by repository)
        # Convert metrics mapping to list-of-dicts for artifact storage
        metrics_list = [{k: v} for k, v in metrics.items()]

        self._save_model(
            command.inverse_decision_mapper_params.model_dump(),
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
            metrics_list,
        )

        # Step 6: Visualize results if a visualizer is provided
        self._visualize_results(
            objectives_train_norm,
            objectives_val_norm,
            decisions_train,
            decisions_val,
            inverse_decision_mapper,
            decisions_normalizer,
        )

        self._logger.log_info("Interpolator training workflow completed.")

    def _initialize_components(
        self,
        model_params_config: Any,
        objectives_normalizer_config: Any,
        decisions_normalizer_config: Any,
    ) -> tuple[BaseInverseDecisionMapper, BaseNormalizer, BaseNormalizer]:
        """Initializes components using their respective factories."""
        inverse_decision_mapper: BaseInverseDecisionMapper = (
            self._inverse_decision_factory.create(params=model_params_config)
        )
        objectives_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=objectives_normalizer_config
        )
        decisions_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=decisions_normalizer_config
        )
        self._logger.log_info("All components initialized.")
        return (
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
        )

    def _prepare_data(
        self,
        test_size: float,
        random_state: int,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> tuple[Any, Any, Any, Any]:
        """Loads, splits, and normalizes the data."""
        raw_data = self._pareto_data_repo.load(filename="pareto_data")
        self._logger.log_info("Raw Pareto data loaded.")

        (
            objectives_train,
            objectives_val,
            decisions_train,
            decisions_val,
        ) = train_test_split(
            raw_data.pareto_front,
            raw_data.pareto_set,
            test_size=test_size,
            random_state=random_state,
        )
        self._logger.log_info(
            f"Data split into training ({len(objectives_train)} samples) and validation ({len(objectives_val)} samples) sets."
        )

        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_val_norm = objectives_normalizer.transform(objectives_val)

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)

        self._logger.log_info("Data normalized.")

        return (
            objectives_train_norm,
            objectives_val_norm,
            decisions_train_norm,
            decisions_val,
        )

    def _train_model(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives_train_norm: Any,
        decisions_train_norm: Any,
    ) -> None:
        """Fits the model on the training data."""
        inverse_decision_mapper.fit(
            objectives=objectives_train_norm, decisions=decisions_train_norm
        )
        self._logger.log_info("Inverse decision mapper model fitted on training data.")

    def _validate_model(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        decisions_normalizer: BaseNormalizer,
        objectives_val_norm: Any,
        decisions_val: Any,
        metric_configs: list[Any],
    ) -> dict[str, Any]:
        """Predicts and calculates validation metrics for all specified configurations."""
        decisions_pred_val_norm = inverse_decision_mapper.predict(objectives_val_norm)

        # Handle probabilistic predictors that return samples with shape
        # (n_samples, n_points, n_decisions). Reduce to (n_points, n_decisions).
        if (
            isinstance(decisions_pred_val_norm, np.ndarray)
            and decisions_pred_val_norm.ndim >= 3
        ):
            decisions_pred_val_norm = decisions_pred_val_norm.mean(axis=0)
        elif (
            isinstance(decisions_pred_val_norm, np.ndarray)
            and decisions_pred_val_norm.ndim == 1
        ):
            decisions_pred_val_norm = decisions_pred_val_norm.reshape(-1, 1)

        decisions_pred_val = decisions_normalizer.inverse_transform(
            decisions_pred_val_norm
        )

        metrics: dict[str, Any] = {}
        for config in metric_configs:
            cfg = config.model_dump() if hasattr(config, "model_dump") else config
            validation_metric: BaseValidationMetric = self._metric_factory.create(
                config=cfg
            )
            score = validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val
            )
            metrics[validation_metric.name] = score

        self._logger.log_metrics(f"Validation Metrics: {metrics}")
        return metrics

    def _save_model(
        self,
        model_params: dict[str, Any],
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
        metrics: list[dict[str, Any]],
    ) -> None:
        """Constructs and saves the final model entity. The repository will assign version_number."""
        trained_model_artifact = ModelArtifact(
            parameters=model_params,
            inverse_decision_mapper=inverse_decision_mapper,
            metrics=metrics,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )
        self._trained_model_repository.save(trained_model_artifact)
        self._logger.log_info("Model artifact saved to repository.")

    def _visualize_results(
        self,
        objectives_train_norm: Any,
        objectives_val_norm: Any,
        decisions_train_norm: Any,
        decisions_val: Any,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        decisions_normalizer: BaseNormalizer,
    ) -> None:
        """Generates plots if a visualizer is available."""
        if self._visualizer:
            decisions_pred_val_norm = inverse_decision_mapper.predict(
                objectives_val_norm
            )

            if (
                isinstance(decisions_pred_val_norm, np.ndarray)
                and decisions_pred_val_norm.ndim >= 3
            ):
                decisions_pred_val_norm = decisions_pred_val_norm.mean(axis=0)
            elif (
                isinstance(decisions_pred_val_norm, np.ndarray)
                and decisions_pred_val_norm.ndim == 1
            ):
                decisions_pred_val_norm = decisions_pred_val_norm.reshape(-1, 1)

            decisions_pred_val = decisions_normalizer.inverse_transform(
                decisions_pred_val_norm
            )

            self._visualizer.plot(
                objectives_train=objectives_train_norm,
                objectives_val=objectives_val_norm,
                decisions_train=decisions_train_norm,
                decisions_val=decisions_val,
                decisions_pred_val=decisions_pred_val,
            )
            self._logger.log_info("Plots generated.")
