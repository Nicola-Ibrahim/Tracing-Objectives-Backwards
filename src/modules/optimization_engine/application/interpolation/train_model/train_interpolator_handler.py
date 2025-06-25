from datetime import datetime

from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.entities.interpolator_model import InterpolatorModel
from ....domain.interpolation.interfaces.base_logger import BaseLogger
from ....domain.interpolation.interfaces.base_metric import BaseValidationMetric
from ....domain.interpolation.interfaces.base_normalizer import BaseNormalizer
from ....domain.interpolation.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ....domain.services.training_service import DecisionMapperTrainingService
from ....infrastructure.inverse_decision_mappers.factory import (
    InverseDecisionMapperFactory,
)
from .train_interpolator_command import TrainInterpolatorCommand


class TrainInterpolatorCommandHandler:
    """
    Handler for the TrainInterpolatorCommand.
    Orchestrates the interpolator training, validation, logging, and persistence processes.
    Dependencies are injected via the constructor.
    """

    def __init__(
        self,
        pareto_data_archiver: BaseParetoDataRepository,
        x_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        inverse_decision_factory: InverseDecisionMapperFactory,
        logger: BaseLogger,
        decision_mapper_training_service: DecisionMapperTrainingService,
        validation_metric_calculator: BaseValidationMetric,
        trained_model_repository: BaseInterpolationModelRepository,
    ):
        self._pareto_data_archiver = pareto_data_archiver
        self._x_normalizer = x_normalizer
        self._y_normalizer = y_normalizer
        self._inverse_decision_factory = inverse_decision_factory
        self._logger = logger
        self._decsion_mapper_training_service = decision_mapper_training_service
        self._validation_metric_calculator = validation_metric_calculator
        self._trained_model_repository = trained_model_repository

    def handle(self, command: TrainInterpolatorCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.

        Args:
            command (TrainInterpolatorCommand): The Pydantic command containing
                                               all necessary training parameters and metadata.
        """
        print(
            f"Handling TrainInterpolatorCommand for model '{command.interpolator_conceptual_name}'..."
        )

        # 1. Create the unfitted interpolator instance dynamically using the factory
        unfitted_inverse_decision_mapper = self._inverse_decision_factory.create(
            interpolator_type=command.type,
            params=command.params,
        )

        # 2. Load raw data using the injected archiver
        raw_data = self._pareto_data_archiver.load(filename=command.data_file_name)
        X_data = raw_data.pareto_set
        Y_data = raw_data.pareto_front

        # 3. Delegate training to the domain service
        (
            fitted_inverse_decision_mapper,
            X_val_norm,
            y_true_val,
            x_normalizer_instance,
            y_normalizer_instance,
        ) = self._decsion_mapper_training_service.train(
            interpolator_instance=unfitted_inverse_decision_mapper,
            X_data=X_data,
            Y_data=Y_data,
            x_normalizer_class=self._x_normalizer.__class__,
            y_normalizer_class=self._y_normalizer.__class__,
            test_size=command.test_size,
            random_state=command.random_state,
        )

        # 4. Delegate prediction to the domain service using the fitted mapper and normalized validation data
        y_pred_val = self._decsion_mapper_training_service.predict(
            fitted_interpolator_instance=fitted_inverse_decision_mapper,
            X_query_norm=X_val_norm,
            y_normalizer_instance=y_normalizer_instance,  # Use the fitted normalizer instance
        )

        # 5. Calculate validation metrics using the injected metric service
        mse = self._validation_metric_calculator.calculate(
            y_true=y_true_val, y_pred=y_pred_val
        )

        # 6. Construct the InterpolatorModel entity with all its metadata
        trained_interpolator_model = InterpolatorModel(
            name=command.interpolator_conceptual_name,
            interpolator_type=command.type.value,
            parameters=command.params,
            inverse_decision_mapper=fitted_inverse_decision_mapper,  # Use the fitted mapper
            metrics={"mse": mse},
            trained_at=datetime.now(),
            training_data_identifier=(
                command.training_data_identifier
                if command.training_data_identifier
                else command.data_source_path
            ),
            description=command.description,
            notes=command.notes,
            collection=command.collection,
        )

        # 7. Log validation metrics
        self._logger.log_metrics(trained_interpolator_model.metrics)

        # 8. Log the InterpolatorModel entity
        self._logger.log_model(model=trained_interpolator_model)

        # 9. Save the InterpolatorModel entity to the repository
        self._trained_model_repository.save(trained_interpolator_model)

        print(
            f"Successfully trained and saved model '{trained_interpolator_model.name}' "
            f"(ID: {trained_interpolator_model.id}) trained at {trained_interpolator_model.trained_at}"
        )
