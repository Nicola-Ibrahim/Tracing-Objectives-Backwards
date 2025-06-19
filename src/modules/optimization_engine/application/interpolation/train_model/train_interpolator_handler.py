from ....domain.interpolation.interfaces.base_logger import BaseLogger
from ....domain.interpolation.interfaces.base_metric import BaseValidationMetric
from ....domain.interpolation.interfaces.base_normalizer import BaseNormalizer
from ....domain.interpolation.interfaces.base_repository import (
    BaseTrainedModelRepository,
)
from ....domain.paretos.interfaces.base_archiver import BaseParetoArchiver
from ....domain.services.training_service import TrainingService
from ....infrastructure.interpolators.factory import InterpolatorFactory
from .train_interpolator_command import TrainInterpolatorCommand


class TrainInterpolatorCommandHandler:
    """
    Handler for the TrainInterpolatorCommand.
    Executes the interpolator training and validation process.
    Dependencies (archiver, normalizer class, interpolator factory, logger, training service, model repository)
    are injected via the constructor.
    """

    def __init__(
        self,
        pareto_data_archiver: BaseParetoArchiver,
        normalizer: BaseNormalizer,
        interpolator_factory: InterpolatorFactory,
        logger: BaseLogger,
        interpolator_trainer: TrainingService,
        validation_metric: BaseValidationMetric,
        trained_model_repository: BaseTrainedModelRepository,
    ):
        self._pareto_data_archiver = pareto_data_archiver
        self._x_normalizer_class = normalizer
        self._y_normalizer_class = normalizer
        self._interpolator_factory = interpolator_factory
        self._logger = logger
        self._interpolator_trainer = interpolator_trainer
        self._validation_metric = validation_metric
        self._trained_model_repository = trained_model_repository

    def handle(self, command: TrainInterpolatorCommand) -> None:
        """
        Executes the training workflow for a given interpolator using the command's data.

        Args:
            command (TrainInterpolatorCommand): The Pydantic command containing
                                               all necessary training parameters.
        """

        # 1. Create the interpolator instance dynamically using the factory
        interpolator_instance = self._interpolator_factory.create_interpolator(
            interpolator_type=command.type,
            params=command.config,
        )

        # 2. Load raw data using the injected archiver
        raw_data = self._pareto_data_archiver.load(filename=command.data_source_path)
        X_data = raw_data.pareto_set
        Y_data = raw_data.pareto_front

        # 3. Delegate training and prediction to the domain service
        y_true_val, y_pred_val, trained_interpolator_model = (
            self._interpolator_trainer.train_and_predict(
                interpolator_instance=interpolator_instance,
                interpolator_name=command.interpolator_name,
                interpolator_type=command.type,
                interpolator_params=command.config,
                X_data=X_data,
                Y_data=Y_data,
                x_normalizer_class=self._x_normalizer_class,
                y_normalizer_class=self._y_normalizer_class,
                test_size=command.test_size,
                random_state=command.random_state,
            )
        )

        # 4. Calculate validation metrics using the injected metric service
        mse = self._validation_metric.calculate(y_true=y_true_val, y_pred=y_pred_val)

        # 5. Enrich the InterpolatorModel entity with calculated metrics
        trained_interpolator_model.metrics = {
            "mse": mse
        }  # Assign the calculated metrics

        # 6. Log validation metrics using the logger
        self._logger.log_metrics(trained_interpolator_model.metrics)

        # 7. Log the InterpolatorModel entity's fitted instance and its metadata using the logger
        self._logger.log_model(
            model=trained_interpolator_model.fitted_interpolator,  # The actual Python object to save
            name=trained_interpolator_model.name,
            model_type=trained_interpolator_model.interpolator_type,  # This is already a string from enum.value
            description=trained_interpolator_model.description,
            parameters=trained_interpolator_model.parameters.model_dump(),  # Pydantic model to dict
            metrics=trained_interpolator_model.metrics,
            notes=trained_interpolator_model.notes,
            collection_name=trained_interpolator_model.collection,
        )

        # 8. Save the InterpolatorModel entity to the repository
        self._trained_model_repository.save(trained_interpolator_model)

        print(
            f"Successfully trained and saved model '{trained_interpolator_model.name}' (ID: {trained_interpolator_model.id})"
        )
