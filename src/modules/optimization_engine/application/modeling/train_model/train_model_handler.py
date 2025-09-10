from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ....domain.modeling.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from ...factories.estimator import EstimatorFactory
from ...factories.mertics import MetricFactory
from ...services.estimator_trainers import (
    CrossValidationTrainer,
    DeterministicModelTrainer,
    ProbabilisticModelTrainer,
)
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
        processed_data_repository: BaseDatasetRepository,
        model_repository: BaseModelArtifactRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        metric_factory: MetricFactory,
    ) -> None:
        self._processed_data_repository = processed_data_repository
        self._estimator_factory = estimator_factory
        self._logger = logger
        self._model_repository = model_repository
        self._metric_factory = metric_factory

    # --------------------- PUBLIC ENTRY ---------------------

    def execute(self, command: TrainModelCommand) -> None:
        """
        Executes the training workflow for a given command.
        Unpacks command attributes to pass only necessary data to sub-methods.
        """
        processed_dataset: ProcessedDataset = self._processed_data_repository.load(
            filename="dataset"
        )

        X_train = processed_dataset.X_train
        y_train = processed_dataset.y_train
        X_test = processed_dataset.X_test
        y_test = processed_dataset.y_test
        X_normalizer = processed_dataset.X_normalizer
        y_normalizer = processed_dataset.y_normalizer

        # Unpack command attributes once at the highest level
        estimator_params = command.estimator_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.estimator_performance_metric_configs
        ]
        random_state = command.random_state
        cv_splits = command.cv_splits
        tune_param_name = command.tune_param_name
        tune_param_range = command.tune_param_range
        learning_curve_steps = command.learning_curve_steps

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        # 1) Train and evaluate model based on command
        parameters = {**estimator.to_dict(), "type": estimator.type}

        if tune_param_name and tune_param_range:
            self._logger.log_info("Starting hyperparameter tuning workflow.")
            artifact = CrossValidationTrainer().search(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                param_name=tune_param_name,
                param_range=tune_param_range,
                metrics=validation_metrics,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                parameters=parameters,
                random_state=random_state,
                cv=cv_splits,
            )
            self._logger.log_info("Hyperparameter tuning workflow completed.")

        elif cv_splits > 1:
            self._logger.log_info("Starting cross-validation workflow.")
            outcome = CrossValidationTrainer().validate(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                validation_metrics=validation_metrics,
                parameters=parameters,
                n_splits=cv_splits,
                random_state=random_state,
                verbose=False,
            )
            self._logger.log_info("Cross-validation workflow completed.")

        else:
            self._logger.log_info("Starting single train/test split workflow.")
            if isinstance(estimator, ProbabilisticEstimator):
                outcome = ProbabilisticModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                )

            elif isinstance(estimator, DeterministicEstimator):
                outcome = DeterministicModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    learning_curve_steps=learning_curve_steps,
                    metrics=validation_metrics,
                    random_state=random_state,
                )

            self._logger.log_info("Model training (single split) completed.")

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=outcome.estimator,
            train_scores=outcome.train_scores,
            test_scores=outcome.test_scores,
            cv_scores=outcome.cv_scores,
            loss_history=outcome.loss_history.model_dump(),
        )

        # 3) Persist artifact
        self._model_repository.save(artifact)
