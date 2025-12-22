from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.dataset import Dataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.deterministic import DeterministicModelTrainer
from ....domain.modeling.services.probabilistic import ProbabilisticModelTrainer
from ...factories.estimator import EstimatorFactory
from ...factories.metrics import MetricFactory
from .command import TrainInverseModelCommand


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
        self._data_repository = processed_data_repository
        self._model_repository = model_repository
        self._logger = logger
        self._estimator_factory = estimator_factory
        self._metric_factory = metric_factory

    def execute(self, command: TrainInverseModelCommand) -> None:
        dataset: Dataset = self._data_repository.load(name=command.dataset_name)
        if not dataset.processed:
            raise ValueError(
                f"Dataset '{dataset.name}' has no processed data available for training."
            )
        processed_data = dataset.processed

        self._logger.log_info("Training inverse model (objectives ➝ decisions).")

        X_train = processed_data.objectives_train
        y_train = processed_data.decisions_train
        X_test = processed_data.objectives_test
        y_test = processed_data.decisions_test
        mapping_direction = "inverse"

        estimator_params = command.estimator_params.model_dump()
        metric_configs = [
            cfg.model_dump() for cfg in command.estimator_performance_metric_configs
        ]
        random_state = command.random_state
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
            "dataset_name": command.dataset_name,
        }

        self._logger.log_info("Starting single train/test split workflow.")
        if isinstance(estimator, ProbabilisticEstimator):
            fitted_estimator, training_history, metrics = (
                ProbabilisticModelTrainer().train(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    epochs=epochs,
                )
            )

        elif isinstance(estimator, DeterministicEstimator):
            fitted_estimator, training_history, metrics = (
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
        else:  # pragma: no cover - defensive guard for unknown estimator types
            raise TypeError(f"Unsupported estimator type: {type(estimator)!r}")
        self._logger.log_info("Model training (single split) completed.")

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=fitted_estimator,
            metrics=metrics,
            training_history=training_history,
        )
        self._model_repository.save(artifact)
