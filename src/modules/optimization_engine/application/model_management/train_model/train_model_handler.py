from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ....domain.model_management.entities.model_artifact import (
    ModelArtifact,
)
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_management.interfaces.base_ml_mapper import (
    BaseMlMapper,
    DeterministicMlMapper,
    ProbabilisticMlMapper,
)
from ....domain.model_management.interfaces.base_normalizer import BaseNormalizer
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ...factories.mertics import MetricFactory
from ...factories.ml_mapper import (
    MlMapperFactory,
)
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
        inverse_decision_factory: MlMapperFactory,
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
        raw_data: DataModel = self._data_repository.load(filename="pareto_data")

        # Initialize components once
        ml_mapper = self._inverse_decision_factory.create(
            params=command.ml_mapper_params.model_dump()
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
                ml_mapper=ml_mapper,
                raw_data=raw_data,
                validation_metrics=validation_metrics,
            )
        else:
            self._logger.log_info("Starting single train/test split training workflow.")
            self._execute_single_split_workflow(
                command=command,
                ml_mapper=ml_mapper,
                raw_data=raw_data,
                validation_metrics=validation_metrics,
            )

    def _execute_single_split_workflow(
        self,
        command: TrainModelCommand,
        ml_mapper: BaseMlMapper,
        raw_data: DataModel,
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
            objectives_test_norm,
            decisions_train_norm,
            decisions_test_norm,
        ) = self._prepare_data(
            raw_data=raw_data,
            test_size=command.test_size,
            random_state=command.random_state,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )

        self._train_model(ml_mapper, objectives_train_norm, decisions_train_norm)

        # Calculate training metrics
        train_scores = self._validate_model(
            ml_mapper=ml_mapper,
            X=objectives_train_norm,
            y=decisions_train_norm,
            validation_metrics=validation_metrics,
        )
        self._logger.log_info(f"Training scores: {train_scores}")

        # Calculate validation metrics
        test_scores = self._validate_model(
            ml_mapper=ml_mapper,
            X=objectives_test_norm,
            y=decisions_test_norm,
            validation_metrics=validation_metrics,
        )
        self._logger.log_info(f"Validation scores: {test_scores}")

        self._save_model(
            ml_mapper=ml_mapper,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores={},
        )

        if self._visualizer:
            self._visualize_results(
                objectives_train_norm,
                objectives_test_norm,
                decisions_train_norm,
                decisions_test_norm,
                ml_mapper,
                decisions_normalizer,
            )

        self._logger.log_info("Model training workflow completed.")

    def _execute_cv_workflow(
        self,
        command: TrainModelCommand,
        ml_mapper: BaseMlMapper,
        raw_data: DataModel,
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> None:
        """Handles the cross-validation workflow, evaluating performance and saving a final model."""

        # Run the cross-validation service to get evaluation metrics
        self._logger.log_info("Running cross-validation to assess model performance...")
        cv_scores = cross_validate(
            estimator=ml_mapper,
            X=raw_data.historical_objectives,
            y=raw_data.historical_solutions,
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
            raw_data.historical_objectives
        )
        decisions_norm = final_decisions_normalizer.fit_transform(
            raw_data.historical_solutions
        )

        final_mapper_instance = _clone(ml_mapper)
        final_mapper_instance.fit(X=objectives_norm, y=decisions_norm)

        # Calculate training metrics on the full dataset
        train_scores = self._validate_model(
            ml_mapper=final_mapper_instance,
            X=objectives_norm,
            y=decisions_norm,
            validation_metrics=validation_metrics,
        )
        self._logger.log_info(f"Training scores: {train_scores}")

        # Save the final model with the aggregated CV metrics
        self._save_model(
            ml_mapper=final_mapper_instance,
            objectives_normalizer=final_objectives_normalizer,
            decisions_normalizer=final_decisions_normalizer,
            train_scores=train_scores,
            test_scores={},  # No dedicated test set in this workflow
            cv_scores=cv_scores,
        )

        if self._visualizer:
            self._visualize_results(
                X_train=objectives_norm,
                X_test=objectives_norm,
                y_train=raw_data.historical_solutions,
                y_test=raw_data.historical_solutions,
                ml_mapper=final_mapper_instance,
                decisions_normalizer=final_decisions_normalizer,
            )

    def _prepare_data(
        self,
        raw_data: DataModel,
        test_size: float,
        random_state: int,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> tuple[Any, Any, Any, Any]:
        """Loads, splits, and normalizes the data."""
        (
            objectives_train,
            objectives_test,
            decisions_train,
            decisions_test,
        ) = train_test_split(
            raw_data.historical_objectives,
            raw_data.historical_solutions,
            test_size=test_size,
            random_state=random_state,
        )
        self._logger.log_info(
            f"Data split into training ({len(objectives_train)} samples) and validation ({len(objectives_test)} samples) sets."
        )

        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_test_norm = objectives_normalizer.transform(objectives_test)

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)
        decisions_test_norm = decisions_normalizer.transform(decisions_test)

        return (
            objectives_train_norm,
            objectives_test_norm,
            decisions_train_norm,
            decisions_test_norm,
        )

    def _train_model(
        self,
        ml_mapper: BaseMlMapper,
        objectives_train_norm: Any,
        decisions_train_norm: Any,
    ) -> None:
        """Fits the model on the training data."""
        ml_mapper.fit(X=objectives_train_norm, y=decisions_train_norm)
        self._logger.log_info("Inverse decision mapper model fitted on training data.")

    def _validate_model(
        self,
        ml_mapper: BaseMlMapper,
        X: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, Any]:
        """Predicts and calculates validation metrics."""

        if isinstance(ml_mapper, ProbabilisticMlMapper):
            y_pred = ml_mapper.predict(X, mode="mean")

        elif isinstance(ml_mapper, DeterministicMlMapper):
            y_pred = ml_mapper.predict(X)

        metrics_list: dict[str, Any] = {}
        for metric_name, validation_metric in validation_metrics.items():
            score = validation_metric.calculate(y_true=y, y_pred=y_pred)
            metrics_list[metric_name] = score

        return metrics_list

    def _save_model(
        self,
        ml_mapper: BaseMlMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
        train_scores: dict[str, Any],
        test_scores: dict[str, Any],
        cv_scores: dict[str, Any],
    ) -> None:
        """Constructs and saves the final model entity."""

        ml_mapper_params = {
            **ml_mapper.to_dict(),
            "type": ml_mapper.type,
        }
        trained_model_artifact = ModelArtifact(
            parameters=ml_mapper_params,
            ml_mapper=ml_mapper,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )
        self._trained_model_repository.save(trained_model_artifact)
        self._logger.log_info("Model artifact saved to repository.")

    def _visualize_results(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        ml_mapper: BaseMlMapper,
        decisions_normalizer: BaseNormalizer,
    ) -> None:
        """Generates plots if a visualizer is available."""

        decisions_pred_test_norm = ml_mapper.predict(X_test)
        if (
            isinstance(decisions_pred_test_norm, np.ndarray)
            and decisions_pred_test_norm.ndim >= 3
        ):
            decisions_pred_test_norm = decisions_pred_test_norm.mean(axis=0)
        elif (
            isinstance(decisions_pred_test_norm, np.ndarray)
            and decisions_pred_test_norm.ndim == 1
        ):
            decisions_pred_test_norm = decisions_pred_test_norm.reshape(-1, 1)

        decisions_pred_test = decisions_normalizer.inverse_transform(
            decisions_pred_test_norm
        )

        self._visualizer.plot(
            objectives_train=X_train,
            objectives_val=X_test,
            decisions_train=y_train,
            decisions_val=y_test,
            decisions_pred_val=decisions_pred_test,
        )
