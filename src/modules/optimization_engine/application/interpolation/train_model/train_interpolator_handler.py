from datetime import datetime

from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.interpolation.entities.interpolator_model import InterpolatorModel
from ....domain.interpolation.interfaces.base_logger import BaseLogger
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
        pareto_data_repo: BaseParetoDataRepository,
        inverse_decision_factory: InverseDecisionMapperFactory,
        logger: BaseLogger,
        decision_mapper_training_service: DecisionMapperTrainingService,
        trained_model_repository: BaseInterpolationModelRepository,
    ):
        self._pareto_data_repo = pareto_data_repo
        self._inverse_decision_factory = inverse_decision_factory
        self._logger = logger
        self._decsion_mapper_training_service = decision_mapper_training_service
        self._trained_model_repository = trained_model_repository

    def handle(self, command: TrainInterpolatorCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.

        Args:
            command (TrainInterpolatorCommand): The Pydantic command containing
                                               all necessary training parameters and metadata.
        """

        # Create the decision mapper using the factory
        inverse_decision_mapper = self._inverse_decision_factory.create(
            type=command.type,
            params=command.params.model_dump(),
        )

        # Load raw data using the injected archiver
        raw_data = self._pareto_data_repo.load(filename=command.data_file_name)

        # Delegate training to the domain service
        fitted_inverse_decision_mapper, validation_metrics = (
            self._decsion_mapper_training_service.train(
                inverse_decision_mapper=inverse_decision_mapper,
                objectives=raw_data.pareto_front,
                decisions=raw_data.pareto_set,
                test_size=command.test_size,
                random_state=command.random_state,
            )
        )

        # Construct the InterpolatorModel entity with all its metadata
        trained_interpolator_model = InterpolatorModel(
            name=command.model_conceptual_name,
            interpolator_type=command.type.value,
            parameters=command.params,
            inverse_decision_mapper=fitted_inverse_decision_mapper,  # Use the fitted mapper
            metrics=validation_metrics,
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

        # # Log validation metrics
        # self._logger.log_metrics(trained_interpolator_model.metrics)

        # # Log the InterpolatorModel entity
        # self._logger.log_model(model=trained_interpolator_model)

        # # Save the InterpolatorModel entity to the repository
        # self._trained_model_repository.save(trained_interpolator_model)

        # print(
        #     f"Successfully trained and saved model '{trained_interpolator_model.name}' "
        #     f"(ID: {trained_interpolator_model.id}) trained at {trained_interpolator_model.trained_at}"
        # )
