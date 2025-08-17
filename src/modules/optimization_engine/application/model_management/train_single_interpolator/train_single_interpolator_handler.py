from typing import Any

from sklearn.model_selection import train_test_split

from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_management.entities.model_artifact import (
    ModelArtifact,
)
from ....domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ....domain.model_management.interfaces.base_normalizer import BaseNormalizer
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ...factories.inverse_decision_mapper import (
    InverseDecisionMapperFactory,
)
from ....infrastructure.metrics import MetricFactory
from ....infrastructure.normalizers import NormalizerFactory
from .train_single_interpolator_command import TrainSingleInterpolatorCommand


class TrainSingleInterpolatorCommandHandler:
    """
    Handler for the TrainSingleInterpolatorCommand.
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

    def execute(self, command: TrainSingleInterpolatorCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.
        """
        self._logger.log_info("Starting single train/test split interpolator training.")

        # Step 1: Initialize all necessary components based on the command
        (
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
            validation_metric,
        ) = self._initialize_components(command)

        # Step 2: Load and prepare the data
        (
            objectives_train_norm,
            objectives_val_norm,
            decisions_train_norm,
            decisions_val_norm,
            decisions_val,
        ) = self._prepare_data(command, objectives_normalizer, decisions_normalizer)

        # Step 3: Train the model
        self._train_model(
            inverse_decision_mapper, objectives_train_norm, decisions_train_norm
        )

        # Step 4: Validate the model and calculate metrics
        metrics = self._validate_model(
            inverse_decision_mapper,
            decisions_normalizer,
            validation_metric,
            objectives_val_norm,
            decisions_val,
        )

        # Step 5: Save the trained model and its metadata
        self._save_model(
            command,
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
            metrics,
        )

        # Step 6: Visualize results if a visualizer is provided
        self._visualize_results(
            objectives_train_norm,
            objectives_val_norm,
            decisions_train_norm,
            decisions_val_norm,
            inverse_decision_mapper,
        )

        self._logger.log_info("Interpolator training workflow completed.")

    def _initialize_components(
        self, command: TrainSingleInterpolatorCommand
    ) -> tuple[Any, Any, Any, Any]:
        """Initializes components using their respective factories."""
        inverse_decision_mapper = self._inverse_decision_factory.create(
            params=command.params.model_dump()
        )
        objectives_normalizer = self._normalizer_factory.create(
            config=command.objectives_normalizer_config.model_dump()
        )
        decisions_normalizer = self._normalizer_factory.create(
            config=command.decisions_normalizer_config.model_dump()
        )
        validation_metric = self._metric_factory.create(
            config=command.validation_metric_config.model_dump()
        )
        self._logger.log_info("All components initialized.")
        return (
            inverse_decision_mapper,
            objectives_normalizer,
            decisions_normalizer,
            validation_metric,
        )

    def _prepare_data(
        self,
        command: TrainSingleInterpolatorCommand,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
    ) -> tuple[
        Any, Any, Any, Any, Any
    ]:  # Replace Any with actual data types, e.g., np.ndarray
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
            test_size=command.test_size,
            random_state=command.random_state,
        )
        self._logger.log_info(
            f"Data split into training ({len(objectives_train)} samples) and validation ({len(objectives_val)} samples) sets."
        )

        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_val_norm = objectives_normalizer.transform(objectives_val)

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)
        decisions_val_norm = decisions_normalizer.transform(decisions_val)
        self._logger.log_info("Data normalized.")

        return (
            objectives_train_norm,
            objectives_val_norm,
            decisions_train_norm,
            decisions_val_norm,
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
        validation_metric: BaseValidationMetric,
        objectives_val_norm: Any,
        decisions_val: Any,
    ) -> dict[str, Any]:
        """Predicts and calculates validation metrics."""
        decisions_pred_val_norm = inverse_decision_mapper.predict(objectives_val_norm)
        decisions_pred_val = decisions_normalizer.inverse_transform(
            decisions_pred_val_norm
        )
        metrics: dict[str, Any] = {
            validation_metric.name: validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val
            )
        }
        self._logger.log_metrics(f"Validation Metrics: {metrics}")
        return metrics

    def _save_model(
        self,
        command: TrainSingleInterpolatorCommand,
        inverse_decision_mapper: BaseInverseDecisionMapper,
        objectives_normalizer: BaseNormalizer,
        decisions_normalizer: BaseNormalizer,
        metrics: dict[str, Any],
    ) -> None:
        """Constructs and saves the final model entity."""
        trained_model_artifact = ModelArtifact(
            parameters=command.params.model_dump(),
            inverse_decision_mapper=inverse_decision_mapper,
            metrics=metrics,
            version_number=command.version_number,
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
        decisions_val_norm: Any,
        inverse_decision_mapper: Any,
    ) -> None:
        """Generates plots if a visualizer is available."""
        if self._visualizer:
            decisions_pred_val_norm = inverse_decision_mapper.predict(
                objectives_val_norm
            )
            self._visualizer.plot(
                objectives_train=objectives_train_norm,
                objectives_val=objectives_val_norm,
                decisions_train=decisions_train_norm,
                decisions_val=decisions_val_norm,
                decisions_pred_val=decisions_pred_val_norm,
            )
            self._logger.log_info("Plots generated.")
