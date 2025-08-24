from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.entities.pareto_data import ParetoDataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ....domain.model_management.entities.model_artifact import (
    ModelArtifact,
)
from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
    DeterministicInverseDecisionMapper,
    ProbabilisticInverseDecisionMapper,
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
from ...services.cross_validation import _clone, cross_validate
from .train_model_command import TrainModelCommand


class TrainModelCommandHandler:
    """
    Handler for the TrainModelCommand.
    Orchestrates the interpolator training, validation, logging, and persistence processes.
    Dependencies are injected via the constructor.
    """

    def __init__(
        self,
        data_repository: BaseParetoDataRepository,
        inverse_decision_factory: InverseDecisionMapperFactory,
        logger: BaseLogger,
        trained_model_repository: BaseInterpolationModelRepository,
        normalizer_factory: NormalizerFactory,
        metric_factory: MetricFactory,
        visualizer: BaseDataVisualizer | None = None,
    ):
        self._data_repository = data_repository
        self._inverse_decision_factory = inverse_decision_factory
        self._logger = logger
        self._trained_model_repository = trained_model_repository
        self._normalizer_factory = normalizer_factory
        self._validation_metric_factory = metric_factory
        self._visualizer = visualizer

    def execute(self, command: TrainModelCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.
        """
        raw_data: ParetoDataModel = self._data_repository.load(filename="pareto_data")

        # Initialize components once
        inverse_decision_mapper = self._inverse_decision_factory.create(
            params=command.inverse_decision_mapper_params.model_dump()
        )

        validation_metrics = self._validation_metric_factory.create_multiple(
            configs=[
                config.model_dump()
                for config in command.model_performance_metric_configs
            ]
        )
        # Create a dictionary of validation metrics, mapping name to instance
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        # Decide on the workflow
        if command.cv_splits is not None:
            self._logger.log_info("Starting cross-validation training workflow.")
            self._execute_cv_workflow(
                command=command,
                inverse_decision_mapper=inverse_decision_mapper,
                raw_data=raw_data,
                validation_metrics=validation_metrics,
            )
        else:
            self._logger.log_info("Starting single train/test split training workflow.")
            self._execute_single_split_workflow(
                command=command,
                inverse_decision_mapper=inverse_decision_mapper,
                raw_data=raw_data,
                validation_metrics=validation_metrics,
            )

    def _execute_single_split_workflow(
        self,
        command: TrainModelCommand,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        raw_data: ParetoDataModel,
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> None:
        """Handles the original single train/test split workflow."""
        objectives_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.objectives_normalizer_config.model_dump()
        )
        decisions_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.decisions_normalizer_config.model_dump()
        )

        (
            objectives_train_norm,
            objectives_val_norm,
            decisions_train,
            decisions_val,
        ) = self._prepare_data(
            raw_data,
            command.test_size,
            command.random_state,
            objectives_normalizer,
            decisions_normalizer,
        )

        self._train_model(
            inverse_decision_mapper, objectives_train_norm, decisions_train
        )

        train_scores = self._validate_model(
            inverse_decision_mapper=inverse_decision_mapper,
            decisions_normalizer=decisions_normalizer,
            objectives_val_norm=objectives_val_norm,
            decisions_val=decisions_val,
            validation_metrics=validation_metrics,
        )

        self._save_model(
            model_params=command.inverse_decision_mapper_params.model_dump(),
            inverse_decision_mapper=inverse_decision_mapper,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
            train_scores=train_scores,
            cv_scores={},
        )

        self._visualize_results(
            objectives_train_norm,
            objectives_val_norm,
            decisions_train,
            decisions_val,
            inverse_decision_mapper,
            decisions_normalizer,
        )

        self._logger.log_info("Model training workflow completed.")

    def _execute_cv_workflow(
        self,
        command: TrainModelCommand,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        raw_data: ParetoDataModel,
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> None:
        """Handles the cross-validation workflow, evaluating performance and saving a final model."""

        # Run the cross-validation service to get evaluation metrics
        self._logger.log_info("Running cross-validation to assess model performance...")

        cv_scores = cross_validate(
            estimator=inverse_decision_mapper,
            X=raw_data.pareto_front,
            y=raw_data.pareto_set,
            validation_metrics=validation_metrics,
            n_splits=command.cv_splits,
            random_state=command.random_state,
            verbose=False,
        )

        # Retrain a final model on the full dataset
        self._logger.log_info("Retraining a final model on the full dataset...")
        final_objectives_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.objectives_normalizer_config.model_dump()
        )
        final_decisions_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.decisions_normalizer_config.model_dump()
        )

        objectives_norm = final_objectives_normalizer.fit_transform(
            raw_data.pareto_front
        )
        decisions_norm = final_decisions_normalizer.fit_transform(raw_data.pareto_set)

        final_mapper_instance = _clone(inverse_decision_mapper)
        final_mapper_instance.fit(X=objectives_norm, y=decisions_norm)

        train_scores = self._validate_model(
            inverse_decision_mapper=final_mapper_instance,
            decisions_normalizer=final_decisions_normalizer,
            objectives_val_norm=objectives_norm,
            decisions_val=raw_data.pareto_set,
            validation_metrics=validation_metrics,
        )

        # Save the final model with the aggregated CV metrics
        self._save_model(
            model_params=command.inverse_decision_mapper_params.model_dump(),
            inverse_decision_mapper=final_mapper_instance,
            objectives_normalizer=final_objectives_normalizer,
            decisions_normalizer=final_decisions_normalizer,
            train_scores=train_scores,
            cv_scores=cv_scores,
        )

        self._visualize_results(
            objectives_train_norm=objectives_norm,
            objectives_val_norm=objectives_norm,
            decisions_train=raw_data.pareto_set,
            decisions_val=raw_data.pareto_set,
            inverse_decision_mapper=final_mapper_instance,
            decisions_normalizer=final_decisions_normalizer,
        )

    def _prepare_data(
        self,
        raw_data: ParetoDataModel,
        test_size: float,
        random_state: int,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> tuple[Any, Any, Any, Any]:
        """Loads, splits, and normalizes the data."""
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
        inverse_decision_mapper.fit(X=objectives_train_norm, y=decisions_train_norm)
        self._logger.log_info("Inverse decision mapper model fitted on training data.")

    def _validate_model(
        self,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        decisions_normalizer: BaseNormalizer,
        objectives_val_norm: Any,
        decisions_val: Any,
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, Any]:
        """Predicts and calculates validation metrics."""

        if isinstance(inverse_decision_mapper, ProbabilisticInverseDecisionMapper):
            decisions_pred_val_norm = inverse_decision_mapper.predict(
                objectives_val_norm, mode="mean"
            )

        elif isinstance(inverse_decision_mapper, DeterministicInverseDecisionMapper):
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

        metrics_list: dict[str, Any] = {}
        for metric_name, validation_metric in validation_metrics.items():
            score = validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val
            )
            metrics_list[metric_name] = score

        return metrics_list

    def _save_model(
        self,
        model_params: dict[str, Any],
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
        train_scores: dict[str, list[float]],
        cv_scores: dict[str, list[float]],
    ) -> None:
        """Constructs and saves the final model entity."""

        inverse_decision_mapper_attrs = {
            **inverse_decision_mapper.to_dict(),
            "type": inverse_decision_mapper.type,
        }

        trained_model_artifact = ModelArtifact(
            parameters=inverse_decision_mapper_attrs,
            inverse_decision_mapper=inverse_decision_mapper,
            train_scores=train_scores,
            cv_scores=cv_scores,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )
        self._trained_model_repository.save(trained_model_artifact)
        self._logger.log_info("Model artifact saved to repository.")

    def _visualize_results(
        self,
        objectives_train_norm: Any,
        objectives_val_norm: Any,
        decisions_train: Any,
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
                decisions_train=decisions_train,
                decisions_val=decisions_val,
                decisions_pred_val=decisions_pred_val,
            )
