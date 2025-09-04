from ....domain.analysis.interfaces.base_visualizer import BaseDataVisualizer
from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ....domain.model_management.entities.model_artifact import ModelArtifact
from ....domain.model_management.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ....domain.model_management.interfaces.base_logger import BaseLogger
from ....domain.model_management.interfaces.base_repository import (
    BaseInterpolationModelRepository,
)
from ...factories.estimator import EstimatorFactory
from ...factories.mertics import MetricFactory
from ...factories.normalizer import NormalizerFactory
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
        data_repository: BaseParetoDataRepository,
        model_repository: BaseInterpolationModelRepository,
        logger: BaseLogger,
        estimator_factory: EstimatorFactory,
        normalizer_factory: NormalizerFactory,
        metric_factory: MetricFactory,
        visualizer: BaseDataVisualizer,
    ) -> None:
        self._data_repository = data_repository
        self._estimator_factory = estimator_factory
        self._logger = logger
        self._model_repository = model_repository
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
        estimator_params = command.estimator_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.estimator_performance_metric_configs
        ]
        normalizer_config = command.normalizer_config.model_dump()
        test_size = command.test_size
        random_state = command.random_state
        cv_splits = command.cv_splits
        tune_param_name = command.tune_param_name
        tune_param_range = command.tune_param_range

        estimator = self._estimator_factory.create(params=estimator_params)
        validation_metrics = self._metric_factory.create_multiple(
            configs=metric_configs
        )
        validation_metrics = {metric.name: metric for metric in validation_metrics}

        # 1) Build normalizers
        objectives_normalizer = self._normalizer_factory.create(
            config=normalizer_config
        )
        decisions_normalizer = self._normalizer_factory.create(config=normalizer_config)

        # 2) Train and evaluate model based on command
        parameters = {**estimator.to_dict(), "type": estimator.type}

        if tune_param_name and tune_param_range:
            self._logger.log_info("Starting hyperparameter tuning workflow.")
            artifact = CrossValidationTrainer().search(
                estimator=estimator,
                X=raw_data.historical_objectives,
                y=raw_data.historical_solutions,
                param_name=tune_param_name,
                param_range=tune_param_range,
                metrics=validation_metrics,
                X_normalizer=decisions_normalizer,
                y_normalizer=objectives_normalizer,
                parameters=parameters,
                test_size=test_size,
                random_state=random_state,
                cv=cv_splits,
            )
            self._logger.log_info("Hyperparameter tuning workflow completed.")

        elif cv_splits > 1:
            self._logger.log_info("Starting cross-validation workflow.")
            outcome = CrossValidationTrainer().validate(
                estimator=estimator,
                X=raw_data.historical_objectives,
                y=raw_data.historical_solutions,
                X_normalizer=decisions_normalizer,
                y_normalizer=objectives_normalizer,
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
                    X=raw_data.historical_objectives,
                    y=raw_data.historical_solutions,
                    X_normalizer=decisions_normalizer,
                    y_normalizer=objectives_normalizer,
                    test_size=test_size,
                    random_state=random_state,
                )

            elif isinstance(estimator, DeterministicEstimator):
                outcome = DeterministicModelTrainer().train(
                    estimator=estimator,
                    X=raw_data.historical_objectives,
                    y=raw_data.historical_solutions,
                    X_normalizer=decisions_normalizer,
                    y_normalizer=objectives_normalizer,
                    learning_curve_steps=50,
                    metrics=validation_metrics,
                    random_state=random_state,
                    test_size=test_size,
                )

            self._logger.log_info("Model training (single split) completed.")

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=outcome.estimator,
            X_normalizer=outcome.X_normalizer,
            y_normalizer=outcome.y_normalizer,
            train_scores=outcome.train_scores,
            test_scores=outcome.test_scores,
            cv_scores={},
            loss_history=outcome.loss_history.model_dump(),
        )

        # 3) Persist artifact
        self._model_repository.save(artifact)

        # TODO: trying to pass only normalized training data

        self._visualizer.plot(
            data={
                "estimator": outcome.estimator,
                "X_train": outcome.X_train,
                "y_train": outcome.y_train,
                "X_test": outcome.X_test,
                "y_test": outcome.y_test,
                "X_normalizer": outcome.X_normalizer,
                "y_normalizer": outcome.y_normalizer,
                "non_linear": False,  # or True to try UMAP if installed
                "n_samples": 300,
                "title": f"Fitted {type(artifact.estimator).__name__}",
            }
        )
