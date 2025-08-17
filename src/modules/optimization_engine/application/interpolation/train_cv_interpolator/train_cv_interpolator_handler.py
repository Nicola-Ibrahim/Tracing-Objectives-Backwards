from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.entities.interpolator_model import InterpolatorModel
from ....domain.interpolation.interfaces.base_logger import BaseLogger
from ....domain.interpolation.interfaces.base_metric import BaseValidationMetric
from ....domain.interpolation.interfaces.base_normalizer import (
    BaseNormalizer,
)
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ....infrastructure.inverse_decision_mappers.factory import (
    InverseDecisionMapperFactory,
)
from ....infrastructure.metrics import MetricFactory
from ....infrastructure.normalizers import NormalizerFactory
from .train_cv_interpolator_command import (
    TrainCvInterpolatorCommand,
)


class TrainCvInterpolatorCommandHandler:
    """
    Handler for the TrainCvInterpolatorCommand.
    Orchestrates the interpolator training, validation, logging, and persistence processes
    using K-fold cross-validation.
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

    def execute(self, command: TrainCvInterpolatorCommand) -> None:
        """
        Executes the cross-validation training workflow for a given interpolator.
        """
        self._logger.log_info(
            f"Starting K-Fold Cross-Validation with {command.cross_validation_config.n_splits} folds."
        )

        # Step 1: Prepare data and initialize core components
        full_objectives, full_decisions = self._prepare_data()
        (
            validation_metric,
            objectives_normalizer_global,
            decisions_normalizer_global,
        ) = self._initialize_components(command)

        # Step 2: Conduct cross-validation and collect fold metrics
        cv_metrics, fold_scores = self._run_cross_validation_procedure(
            command, full_objectives, full_decisions, validation_metric
        )

        # Step 3: Train final production model on the entire dataset
        final_model, final_normalizers = self._train_final_model(
            command,
            full_objectives,
            full_decisions,
            objectives_normalizer_global,
            decisions_normalizer_global,
        )

        # Step 4: Persist and log the final model with its performance
        self._persist_and_log_model_artifact(
            command, final_model, final_normalizers, cv_metrics, fold_scores
        )

        self._logger.log_info("K-Fold Cross-Validation workflow completed.")

    def _prepare_data(self) -> tuple[Any, Any]:
        """Loads the full dataset from the repository."""
        raw_data = self._pareto_data_repo.load(filename="pareto_data")
        full_objectives = raw_data.pareto_front
        full_decisions = raw_data.pareto_set

        self._logger.log_info("Raw Pareto data loaded.")
        return full_objectives, full_decisions

    def _initialize_components(
        self, command: TrainCvInterpolatorCommand
    ) -> tuple[BaseValidationMetric, BaseNormalizer, BaseNormalizer]:
        """Instantiates the performance metric and normalizers based on the command."""
        validation_metric = self._metric_factory.create(
            config=command.validation_metric_config.model_dump()
        )
        objectives_normalizer_global = self._normalizer_factory.create(
            config=command.objectives_normalizer_config.model_dump()
        )
        decisions_normalizer_global = self._normalizer_factory.create(
            config=command.decisions_normalizer_config.model_dump()
        )
        return (
            validation_metric,
            objectives_normalizer_global,
            decisions_normalizer_global,
        )

    def _run_cross_validation_procedure(
        self,
        command: TrainCvInterpolatorCommand,
        full_objectives: Any,
        full_decisions: Any,
        validation_metric: Any,
    ) -> tuple[dict[str, Any], dict[str, list]]:
        """Executes the K-fold validation loop and aggregates all fold scores."""
        k_folder = KFold(
            n_splits=command.cross_validation_config.n_splits,
            shuffle=command.cross_validation_config.shuffle,
            random_state=command.cross_validation_config.random_state,
        )

        fold_scores = defaultdict(list)

        for fold_idx, (train_indices, validation_indices) in enumerate(
            k_folder.split(full_objectives)
        ):
            self._logger.log_info(
                f"Processing Fold {fold_idx + 1}/{command.cross_validation_config.n_splits}"
            )

            # Use dedicated helper method to execute each fold
            self._execute_single_fold(
                command,
                fold_idx,
                train_indices,
                validation_indices,
                full_objectives,
                full_decisions,
                validation_metric,
                fold_scores,
            )

        # Aggregate metrics from all folds
        cv_metrics = self._aggregate_cv_metrics(fold_scores, validation_metric.name)
        return cv_metrics, fold_scores

    def _execute_single_fold(
        self,
        command: TrainCvInterpolatorCommand,
        fold_idx: int,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        full_objectives: Any,
        full_decisions: Any,
        validation_metric: Any,
        fold_scores: defaultdict,
    ) -> None:
        """Trains and evaluates the model for a single cross-validation fold."""
        # Split data for the current fold
        objectives_train, objectives_val = (
            full_objectives[train_indices],
            full_objectives[validation_indices],
        )
        decisions_train, decisions_val = (
            full_decisions[train_indices],
            full_decisions[validation_indices],
        )

        # Create and fit normalizers for the current fold to prevent data leakage
        fold_objectives_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.objectives_normalizer_config.model_dump()
        )
        fold_decisions_normalizer: BaseNormalizer = self._normalizer_factory.create(
            config=command.decisions_normalizer_config.model_dump()
        )
        objectives_train_norm = fold_objectives_normalizer.fit_transform(
            objectives_train
        )
        decisions_train_norm = fold_decisions_normalizer.fit_transform(decisions_train)

        # Instantiate and train the model for this fold
        inverse_decision_mapper = self._inverse_decision_factory.create(
            params=command.params.model_dump(),
        )
        inverse_decision_mapper.fit(
            objectives=objectives_train_norm, decisions=decisions_train_norm
        )

        # Evaluate the model on the validation set
        objectives_val_norm = fold_objectives_normalizer.transform(objectives_val)
        decisions_norm_pred = inverse_decision_mapper.predict(objectives_val_norm)
        decisions_pred = fold_decisions_normalizer.inverse_transform(
            decisions_norm_pred
        )

        # Calculate and record the metric
        score = validation_metric.calculate(y_true=decisions_val, y_pred=decisions_pred)
        fold_scores[validation_metric.name].append(score)
        self._logger.log_info(f"Fold {fold_idx + 1} score: {score:.4f}")

        # Generate plots for the current fold if enabled
        if command.generate_plots_per_fold and self._visualizer:
            self._logger.log_info(f"Generating plots for Fold {fold_idx + 1}...")
            self._visualizer.plot(
                objectives_train=objectives_train_norm,
                objectives_val=objectives_val_norm,
                decisions_train=decisions_train_norm,
                decisions_val=decisions_val,
                decisions_pred_val=decisions_pred,
            )
            self._logger.log_info(f"Plots for Fold {fold_idx + 1} generated.")

    def _aggregate_cv_metrics(
        self, fold_scores: defaultdict, metric_name: str
    ) -> dict[str, Any]:
        """Calculates and formats final aggregated CV metrics."""
        cv_metrics = {}
        values = fold_scores[metric_name]
        mean_score = np.mean(values)
        std_score = np.std(values)

        cv_metrics[f"cv_mean_{metric_name}"] = float(mean_score)
        cv_metrics[f"cv_std_{metric_name}"] = float(std_score)

        # This structure is great for storing the results, it's a good practice to separate the final metrics for logging from the detailed results
        detailed_cv_results = {
            metric_name: {
                "folds": [float(v) for v in values],
                "mean": float(mean_score),
                "std": float(std_score),
            }
        }

        self._logger.log_info(
            f"Aggregated CV {metric_name}: Mean={mean_score:.4f}, Std={std_score:.4f}"
        )
        return detailed_cv_results

    def _train_final_model(
        self,
        command: TrainCvInterpolatorCommand,
        full_objectives: Any,
        full_decisions: Any,
        objectives_normalizer_global: Any,
        decisions_normalizer_global: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Trains the final production model on the entire dataset."""
        self._logger.log_info("Training final model on full dataset for deployment.")

        # Fit normalizers on the full dataset
        norm_full_objectives = objectives_normalizer_global.fit_transform(
            full_objectives
        )
        norm_full_decisions = decisions_normalizer_global.fit_transform(full_decisions)

        # Create and fit the final interpolator
        final_interpolator = self._inverse_decision_factory.create(
            params=command.params.model_dump(),
        )
        final_interpolator.fit(norm_full_objectives, norm_full_decisions)
        self._logger.log_info("Final model trained on full dataset.")

        final_normalizers = {
            "objectives_normalizer": objectives_normalizer_global,
            "decisions_normalizer": decisions_normalizer_global,
        }

        return final_interpolator, final_normalizers

    def _persist_and_log_model_artifact(
        self,
        command: TrainCvInterpolatorCommand,
        final_model: Any,
        final_normalizers: dict[str, Any],
        cv_metrics: dict[str, Any],
        fold_scores: defaultdict,
    ) -> None:
        """Constructs, saves, and logs the final model entity with CV metrics."""
        # Calculate final aggregated metrics for the model entity
        final_metrics_for_entity = {}
        for metric_name, values in fold_scores.items():
            final_metrics_for_entity[f"cv_mean_{metric_name}"] = np.mean(values)
            final_metrics_for_entity[f"cv_std_{metric_name}"] = np.std(values)

        # Construct the final model entity
        trained_interpolator_model = InterpolatorModel(
            parameters=command.params.model_dump(),
            inverse_decision_mapper=final_model,
            metrics=final_metrics_for_entity,
            cross_validation_results=cv_metrics,
            version_number=command.version_number,
            objectives_normalizer=final_normalizers["objectives_normalizer"],
            decisions_normalizer=final_normalizers["decisions_normalizer"],
        )

        # Log the model artifact for version control and tracking
        self._logger.log_model(
            model=trained_interpolator_model.inverse_decision_mapper,
            name=f"{command.params.get('model_type', 'Interpolator')}_v{command.version_number}_cv",
            model_type=command.params.get("model_type", "Unknown"),
            description=f"Interpolator model trained with {command.cross_validation_config.n_splits}-fold CV.",
            parameters=command.params.model_dump(),
            metrics=final_metrics_for_entity,
            notes=f"Cross-validation results for version {command.version_number}.",
            collection_name="cv_interpolator_models",
            step=command.version_number,
        )
        self._logger.log_info("Interpolator model artifact logged.")

        # Save the model entity to the repository
        self._trained_model_repository.save(trained_interpolator_model)
        self._logger.log_info("Interpolator model with CV results saved to repository.")
