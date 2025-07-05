from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.entities.interpolator_model import InterpolatorModel
from ....domain.interpolation.interfaces.base_logger import BaseLogger
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

        Args:
            command (TrainCvInterpolatorCommand): The Pydantic command containing
                                                  all necessary training and CV parameters.
        """
        self._logger.log_info(
            f"Starting K-Fold Cross-Validation with {command.cross_validation_config.n_splits} folds."
        )

        # Load raw data
        raw_data = self._pareto_data_repo.load(filename="pareto_data")
        objectives_full = raw_data.pareto_front
        decisions_full = raw_data.pareto_set

        # Create validation metric
        validation_metric = self._metric_factory.create(
            metric_type=command.validation_metric_config.type,
            **command.validation_metric_config.params,
        )

        kf = KFold(
            n_splits=command.cross_validation_config.n_splits,
            shuffle=command.cross_validation_config.shuffle,
            random_state=command.cross_validation_config.random_state,
        )

        fold_metrics = defaultdict(list)
        cv_results = {}

        # Keep track of the normalizers and the last trained model from a fold,
        # though the final model will be trained on the full dataset.
        objectives_normalizer_final = None
        decisions_normalizer_final = None
        final_inverse_decision_mapper = None

        for fold_idx, (train_index, val_index) in enumerate(kf.split(objectives_full)):
            self._logger.log_info(
                f"Processing Fold {fold_idx + 1}/{command.cross_validation_config.n_splits}"
            )

            objectives_train, objectives_val = (
                objectives_full[train_index],
                objectives_full[val_index],
            )
            decisions_train, decisions_val = (
                decisions_full[train_index],
                decisions_full[val_index],
            )

            # Re-initialize decision mapper for each fold to ensure independent training
            inverse_decision_mapper_fold = self._inverse_decision_factory.create(
                params=command.params.model_dump(),
            )

            # Normalizers must be refitted for each fold's training data to prevent data leakage
            fold_objectives_normalizer = self._normalizer_factory.create(
                normalizer_type=command.objectives_normalizer_config.type,
                **command.objectives_normalizer_config.params,
            )
            fold_decisions_normalizer = self._normalizer_factory.create(
                normalizer_type=command.decisions_normalizer_config.type,
                **command.decisions_normalizer_config.params,
            )

            objectives_train_norm = fold_objectives_normalizer.fit_transform(
                objectives_train
            )
            objectives_val_norm = fold_objectives_normalizer.transform(objectives_val)

            decisions_train_norm = fold_decisions_normalizer.fit_transform(
                decisions_train
            )
            decisions_val_norm = fold_decisions_normalizer.transform(decisions_val)

            # Fit the interpolator instance on normalized data
            inverse_decision_mapper_fold.fit(
                objectives=objectives_train_norm, decisions=decisions_train_norm
            )

            # Predict decision values on the validation set
            decisions_pred_val_norm = inverse_decision_mapper_fold.predict(
                objectives_val_norm
            )

            # Inverse-transform predictions to original scale
            decisions_pred_val = fold_decisions_normalizer.inverse_transform(
                decisions_pred_val_norm
            )

            # Calculate validation metrics for this fold
            fold_metric_value = validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val
            )
            fold_metrics[validation_metric.name].append(fold_metric_value)
            self._logger.log_info(
                f"Fold {fold_idx + 1} {validation_metric.name}: {fold_metric_value:.4f}"
            )

            if command.generate_plots_per_fold and self._visualizer:
                self._logger.log_info(f"Generating plots for Fold {fold_idx + 1}...")
                self._visualizer.plot(
                    raw_pareto_data=type(
                        "RawData",
                        (object,),
                        {
                            "pareto_front": objectives_val,
                            "pareto_set": decisions_val,
                            "decisions_pred_val": decisions_pred_val_norm,  # Normalized predictions for plotting
                            "objectives_val_norm": objectives_val_norm,  # Normalized objectives for plotting
                        },
                    )()
                )
                self._logger.log_info(f"Plots for Fold {fold_idx + 1} generated.")

        # Aggregate CV results
        final_metrics = {}
        for metric_name, values in fold_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            final_metrics[f"cv_mean_{metric_name}"] = float(
                mean_val
            )  # Ensure JSON serializable
            final_metrics[f"cv_std_{metric_name}"] = float(
                std_val
            )  # Ensure JSON serializable
            cv_results[metric_name] = {
                "folds": [float(v) for v in values],  # Ensure JSON serializable
                "mean": float(mean_val),
                "std": float(std_val),
            }
            self._logger.log_info(
                f"Aggregated CV {metric_name}: Mean={mean_val:.4f}, Std={std_val:.4f}"
            )

        # Train a final model on the full dataset for deployment, using normalizers fitted on full data
        self._logger.log_info("Training final model on full dataset for deployment.")
        objectives_normalizer_final = self._normalizer_factory.create(
            normalizer_type=command.objectives_normalizer_config.type,
            **command.objectives_normalizer_config.params,
        )
        decisions_normalizer_final = self._normalizer_factory.create(
            normalizer_type=command.decisions_normalizer_config.type,
            **command.decisions_normalizer_config.params,
        )

        objectives_full_norm = objectives_normalizer_final.fit_transform(
            objectives_full
        )
        decisions_full_norm = decisions_normalizer_final.fit_transform(decisions_full)

        final_inverse_decision_mapper = self._inverse_decision_factory.create(
            params=command.params.model_dump(),
        )
        final_inverse_decision_mapper.fit(objectives_full_norm, decisions_full_norm)
        self._logger.log_info("Final model trained on full dataset.")

        # Construct the InterpolatorModel entity with all its metadata
        trained_interpolator_model = InterpolatorModel(
            parameters=command.params.model_dump(),
            inverse_decision_mapper=final_inverse_decision_mapper,  # The model trained on full data
            metrics=final_metrics,  # Aggregated CV metrics
            cross_validation_results=cv_results,
            version_number=command.version_number,
            objectives_normalizer=objectives_normalizer_final,  # Normalizers from full dataset fit
            decisions_normalizer=decisions_normalizer_final,
        )

        # Log the model artifact using the updated logger method
        self._logger.log_model(
            model=trained_interpolator_model.inverse_decision_mapper,
            name=f"{command.params.get('model_type', 'Interpolator')}_v{command.version_number}_cv",
            model_type=command.params.get("model_type", "Unknown"),
            description=f"Interpolator model trained with {command.cross_validation_config.n_splits}-fold CV.",
            parameters=command.params.model_dump(),
            metrics=final_metrics,
            notes=f"Cross-validation results for version {command.version_number}.",
            collection_name="cv_interpolator_models",  # Example collection name
            step=command.version_number,  # Using version number as step for model logging
        )
        self._logger.log_info("Interpolator model artifact logged.")

        # Save the InterpolatorModel entity to the repository
        self._trained_model_repository.save(trained_interpolator_model)
        self._logger.log_info("Interpolator model with CV results saved to repository.")
