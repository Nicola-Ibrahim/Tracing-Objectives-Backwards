from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ....domain.model_management.entities.model_artifact import ModelArtifact
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
from ...factories.ml_mapper import MlMapperFactory
from ...factories.normalizer import NormalizerFactory
from ...services.cross_validation import _clone, cross_validate
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
        inverse_decision_factory: MlMapperFactory,
        logger: BaseLogger,
        trained_model_repository: BaseInterpolationModelRepository,
        normalizer_factory: NormalizerFactory,
        metric_factory: MetricFactory,
        visualizer: BaseDataVisualizer | None = None,
    ) -> None:
        self._data_repository = data_repository
        self._inverse_decision_factory = inverse_decision_factory
        self._logger = logger
        self._trained_model_repository = trained_model_repository
        self._normalizer_factory = normalizer_factory
        self._metric_factory = metric_factory
        self._visualizer = visualizer

    # --------------------- PUBLIC ENTRY ---------------------

    def execute(self, command: TrainModelCommand) -> None:
        """
        Executes the training workflow for a given command.
        Unpacks command attributes to pass only necessary data to sub-methods.
        """
        raw_data: DataModel = self._data_repository.load(filename="pareto_data")

        # Unpack command attributes once at the highest level
        ml_mapper_params = command.ml_mapper_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.model_performance_metric_configs
        ]
        objectives_normalizer_config = command.objectives_normalizer_config.model_dump()
        decisions_normalizer_config = command.decisions_normalizer_config.model_dump()
        test_size = command.test_size
        random_state = command.random_state
        cv_splits = command.cv_splits

        ml_mapper = self._create_mapper(ml_mapper_params)
        validation_metrics = self._create_metrics(metric_configs)

        if cv_splits:
            self._logger.log_info("Starting cross-validation workflow.")
            self._execute_cv_workflow(
                raw_data,
                ml_mapper,
                validation_metrics,
                objectives_normalizer_config,
                decisions_normalizer_config,
                cv_splits,
                random_state,
            )
        else:
            self._logger.log_info("Starting single train/test split workflow.")
            self._execute_single_split_workflow(
                raw_data,
                ml_mapper,
                validation_metrics,
                objectives_normalizer_config,
                decisions_normalizer_config,
                test_size,
                random_state,
            )

    # --------------------- WORKFLOWS ---------------------

    def _execute_single_split_workflow(
        self,
        raw_data: DataModel,
        ml_mapper: BaseMlMapper,
        validation_metrics: dict[str, BaseValidationMetric],
        objectives_normalizer_config: dict,
        decisions_normalizer_config: dict,
        test_size: float,
        random_state: int,
    ) -> None:
        """Single train/test split workflow."""

        objectives_norm, decisions_norm = self._create_normalizers(
            objectives_normalizer_config, decisions_normalizer_config
        )

        X_train, X_test, y_train, y_test = self._prepare_data(
            raw_data,
            test_size,
            random_state,
            objectives_norm,
            decisions_norm,
        )

        self._fit_model(ml_mapper, X_train, y_train)

        train_scores = self._evaluate(ml_mapper, X_train, y_train, validation_metrics)
        test_scores = self._evaluate(ml_mapper, X_test, y_test, validation_metrics)

        self._logger.log_info(f"Training scores: {train_scores}")
        self._logger.log_info(f"Test scores: {test_scores}")

        self._save_model(
            ml_mapper,
            objectives_norm,
            decisions_norm,
            train_scores,
            test_scores,
            cv_scores={},
        )

        # Prepare data for the learning curve visualizer
        learning_curve_data = {
            "ml_mapper": ml_mapper,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            # Define the sizes of the training subsets to plot
            "train_sizes": np.linspace(
                start=0.1, stop=1.0, num=10, endpoint=False
            ).round(2),
        }

        # Pass the prepared data to the visualizer's plot method
        self._visualizer.plot(data=learning_curve_data)
        # -------------------------------

        self._logger.log_info("Model training (single split) completed.")

    def _execute_cv_workflow(
        self,
        raw_data: DataModel,
        ml_mapper: BaseMlMapper,
        validation_metrics: dict[str, BaseValidationMetric],
        objectives_normalizer_config: dict,
        decisions_normalizer_config: dict,
        cv_splits: int,
        random_state: int,
    ) -> None:
        """Cross-validation workflow with retraining on full dataset."""
        self._logger.log_info("Running cross-validation...")
        cv_scores = cross_validate(
            estimator=ml_mapper,
            X=raw_data.historical_objectives,
            y=raw_data.historical_solutions,
            validation_metrics=validation_metrics,
            n_splits=cv_splits,
            random_state=random_state,
            verbose=False,
        )

        objectives_norm, decisions_norm = self._create_normalizers(
            objectives_normalizer_config, decisions_normalizer_config
        )
        X_norm = objectives_norm.fit_transform(raw_data.historical_objectives)
        y_norm = decisions_norm.fit_transform(raw_data.historical_solutions)

        final_mapper = _clone(ml_mapper)
        self._fit_model(final_mapper, X_norm, y_norm)

        train_scores = self._evaluate(final_mapper, X_norm, y_norm, validation_metrics)
        self._logger.log_info(f"Training scores: {train_scores}")

        self._save_model(
            final_mapper, objectives_norm, decisions_norm, train_scores, {}, cv_scores
        )
        self._visualize(
            X_norm,
            X_norm,
            raw_data.historical_solutions,
            raw_data.historical_solutions,
            final_mapper,
            decisions_norm,
        )

        self._logger.log_info("Model training (cross-validation) completed.")

    # --------------------- HELPERS ---------------------

    def _create_mapper(self, params: dict) -> BaseMlMapper:
        """Factory method to build ML mapper from parameters."""
        return self._inverse_decision_factory.create(params=params)

    def _create_metrics(self, configs: list[dict]) -> dict[str, BaseValidationMetric]:
        """Factory method to build validation metrics from configs."""
        metrics = self._metric_factory.create_multiple(configs=configs)
        return {m.name: m for m in metrics}

    def _create_normalizers(
        self, objectives_normalizer_config: dict, decisions_normalizer_config: dict
    ) -> tuple[BaseNormalizer, BaseNormalizer]:
        """Factory method to create normalizers for objectives & decisions."""
        objectives_normalizer = self._normalizer_factory.create(
            config=objectives_normalizer_config
        )
        decisions_normalizer = self._normalizer_factory.create(
            config=decisions_normalizer_config
        )
        return objectives_normalizer, decisions_normalizer

    def _prepare_data(
        self,
        raw_data: DataModel,
        test_size: float,
        random_state: int,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> tuple[Any, Any, Any, Any]:
        """Splits and normalizes training & test data."""
        X_train, X_test, y_train, y_test = train_test_split(
            raw_data.historical_objectives,
            raw_data.historical_solutions,
            test_size=test_size,
            random_state=random_state,
        )
        self._logger.log_info(
            f"Data split: {len(X_train)} train / {len(X_test)} val samples."
        )

        return (
            objectives_normalizer.fit_transform(X_train),
            objectives_normalizer.transform(X_test),
            decisions_normalizer.fit_transform(y_train),
            decisions_normalizer.transform(y_test),
        )

    def _fit_model(self, ml_mapper: BaseMlMapper, X: Any, y: Any) -> None:
        """Fits ML mapper."""
        ml_mapper.fit(X=X, y=y)
        self._logger.log_info("Model fitted.")

    def _evaluate(
        self,
        ml_mapper: BaseMlMapper,
        X: np.typing.NDArray[np.float64],
        y: np.typing.NDArray[np.float64],
        metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, Any]:
        """Runs predictions and evaluates with given metrics."""
        y_pred = (
            ml_mapper.predict(X, mode="mean")
            if isinstance(ml_mapper, ProbabilisticMlMapper)
            else ml_mapper.predict(X)
        )
        return {
            name: metric.calculate(y_true=y, y_pred=y_pred)
            for name, metric in metrics.items()
        }

    def _save_model(
        self,
        ml_mapper: BaseMlMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
        train_scores: dict[str, Any],
        test_scores: dict[str, Any],
        cv_scores: dict[str, Any],
    ) -> None:
        """Saves trained model as artifact."""
        artifact = ModelArtifact(
            parameters={**ml_mapper.to_dict(), "type": ml_mapper.type},
            ml_mapper=ml_mapper,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )
        self._trained_model_repository.save(artifact)
        self._logger.log_info("Model artifact saved.")

    def _visualize(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        ml_mapper: BaseMlMapper,
        decisions_normalizer: BaseNormalizer,
    ) -> None:
        """Runs visualization if visualizer is provided."""
        if not self._visualizer:
            return

        y_pred_norm = ml_mapper.predict(X_test)
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
