from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.cross_validation import CrossValidationTrainer
from ....domain.modeling.services.deterministic import DeterministicModelTrainer
from ....domain.modeling.services.probabilistic import ProbabilisticModelTrainer
from ...factories.estimator import EstimatorFactory
from ...factories.mertics import MetricFactory
from .train_inverse_model_command import TrainInverseModelCommand


class TrainInverseModelCommandHandler:
    """Train, evaluate, and persist inverse (objectives ➝ decisions) estimators."""

    def __init__(
        self,
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseModelArtifactRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        metric_factory: MetricFactory,
    ) -> None:
        self._processed_data_repository = processed_data_repository
        self._model_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._metric_factory = metric_factory

    def execute(self, command: TrainInverseModelCommand) -> None:
        processed_dataset: ProcessedDataset = self._processed_data_repository.load(
            filename="dataset", variant="processed"
        )
        self._logger.log_info("Training inverse model (objectives ➝ decisions).")

        X_train = processed_dataset.y_train
        y_train = processed_dataset.X_train
        X_test = processed_dataset.y_test
        y_test = processed_dataset.X_test
        X_normalizer = processed_dataset.y_normalizer
        y_normalizer = processed_dataset.X_normalizer
        mapping_direction = "inverse"

        estimator_params = command.estimator_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.estimator_performance_metric_configs
        ]
        random_state = command.random_state
        cv_splits = command.cv_splits
        tune_param_name = command.tune_param_name
        tune_param_range = command.tune_param_range
        learning_curve_steps = command.learning_curve_steps
        epochs = command.epochs

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        parameters = {
            **estimator.to_dict(),
            "type": estimator.type,
            "mapping_direction": mapping_direction,
        }

        if tune_param_name and tune_param_range:
            self._logger.log_info("Starting hyperparameter tuning workflow.")
            fitted_estimator, loss_history, metrics = CrossValidationTrainer().search(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                param_name=tune_param_name,
                param_range=tune_param_range,
                validation_metrics=validation_metrics,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                parameters=parameters,
                random_state=random_state,
                cv=cv_splits,
            )
            self._logger.log_info("Hyperparameter tuning workflow completed.")

        elif cv_splits > 1:
            self._logger.log_info("Starting cross-validation workflow.")
            fitted_estimator, loss_history, metrics = CrossValidationTrainer().validate(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                validation_metrics=validation_metrics,
                epochs=epochs,
                n_splits=cv_splits,
                random_state=random_state,
            )
            self._logger.log_info("Cross-validation workflow completed.")

        else:
            self._logger.log_info("Starting single train/test split workflow.")
            if isinstance(estimator, ProbabilisticEstimator):
                fitted_estimator, loss_history, metrics = (
                    ProbabilisticModelTrainer().train(
                        estimator=estimator,
                        X_train=X_train,
                        y_train=y_train,
                    )
                )
            elif isinstance(estimator, DeterministicEstimator):
                fitted_estimator, loss_history, metrics = (
                    DeterministicModelTrainer().train(
                        estimator=estimator,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        learning_curve_steps=learning_curve_steps,
                        validation_metrics=validation_metrics,
                        random_state=random_state,
                    )
                )
            self._logger.log_info("Model training (single split) completed.")

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=fitted_estimator,
            metrics=metrics,
            loss_history=loss_history,
        )
        self._model_repository.save(artifact)
