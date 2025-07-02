import numpy as np
from sklearn.model_selection import train_test_split

from ....domain.analyzing.interfaces.base_visualizer import BaseDataVisualizer
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

        Args:
            command (TrainSingleInterpolatorCommand): The Pydantic command containing
                                               all necessary training parameters and metadata.
        """
        self._logger.log_info("Starting single train/test split interpolator training.")

        # Create the decision mapper using the factory
        inverse_decision_mapper = self._inverse_decision_factory.create(
            params=command.params.model_dump(),
        )

        objectives_normalizer = self._normalizer_factory.create(
            normalizer_type=command.objectives_normalizer_config.type,
            **command.objectives_normalizer_config.params,
        )

        decisions_normalizer = self._normalizer_factory.create(
            normalizer_type=command.decisions_normalizer_config.type,
            **command.decisions_normalizer_config.params,
        )

        # Create validation metric using the metric factory, based on command config
        validation_metric = self._metric_factory.create(
            metric_type=command.validation_metric_config.type,
            **command.validation_metric_config.params,
        )

        # Load raw data using the injected archiver
        raw_data = self._pareto_data_repo.load(filename="pareto_data")
        self._logger.log_info("Raw Pareto data loaded.")

        # Split data into train and validation sets
        objectives_train, objectives_val, decisions_train, decisions_val = (
            train_test_split(
                raw_data.pareto_front,
                raw_data.pareto_set,
                test_size=command.test_size,
                random_state=command.random_state,
            )
        )
        self._logger.log_info(
            f"Data split into training ({len(objectives_train)} samples) and validation ({len(objectives_val)} samples) sets."
        )

        # Normalize training and validation data
        objectives_train_norm = objectives_normalizer.fit_transform(objectives_train)
        objectives_val_norm = objectives_normalizer.transform(objectives_val)

        decisions_train_norm = decisions_normalizer.fit_transform(decisions_train)
        decisions_val_norm = decisions_normalizer.transform(decisions_val)

        # Fit the interpolator instance on normalized data
        inverse_decision_mapper.fit(
            objectives=objectives_train_norm, decisions=decisions_train_norm
        )
        self._logger.log_info("Inverse decision mapper model fitted on training data.")

        # Predict decision values on the validation set
        decisions_pred_val_norm = inverse_decision_mapper.predict(objectives_val_norm)

        # Inverse-transform predictions to original scale
        decisions_pred_val = decisions_normalizer.inverse_transform(
            decisions_pred_val_norm
        )

        # Calculate validation metrics using the injected metric
        metrics = {
            validation_metric.name: validation_metric.calculate(
                y_true=decisions_val, y_pred=decisions_pred_val
            )
        }
        self._logger.log_metrics(f"Validation Metrics: {metrics}")

        # Construct the InterpolatorModel entity with all its metadata
        trained_interpolator_model = InterpolatorModel(
            parameters=command.params.model_dump(),
            inverse_decision_mapper=inverse_decision_mapper,
            metrics=metrics,
            version_number=command.version_number,
            objectives_normalizer=objectives_normalizer,
            decisions_normalizer=decisions_normalizer,
        )

        # Save the InterpolatorModel entity to the repository
        self._trained_model_repository.save(trained_interpolator_model)
        self._logger.log_info("Interpolator model saved to repository.")

        if self._visualizer:
            # The visualizer expects an object with pareto_front and pareto_set.
            # We pass the validation data and predictions for plotting.
            self._visualizer.plot(
                objectives_train=objectives_train_norm,
                objectives_val=objectives_val_norm,
                decisions_train=decisions_train_norm,
                decisions_val=decisions_val_norm,
                decisions_pred_val=decisions_pred_val_norm,
            )

            self._logger.log_info("Plots generated.")
