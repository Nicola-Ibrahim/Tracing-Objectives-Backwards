from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.processed_dataset import ProcessedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.modeling.entities.model_artifact import ModelArtifact
from ....domain.modeling.interfaces.base_repository import BaseModelArtifactRepository
from ....domain.modeling.services.cross_validation import CrossValidationTrainer
from ...factories.estimator import EstimatorFactory
from ...factories.mertics import MetricFactory
from .command import TrainInverseModelGridSearchCommand


class TrainInverseModelGridSearchCommandHandler:
    """Train, tune, and persist inverse estimators using grid search."""

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

    def execute(self, command: TrainInverseModelGridSearchCommand) -> None:
        processed_dataset: ProcessedDataset = self._processed_data_repository.load(
            filename="dataset", variant="processed"
        )
        self._logger.log_info(
            "Training inverse model with grid search (objectives ➝ decisions)."
        )

        X_train = processed_dataset.y_train
        y_train = processed_dataset.X_train
        X_test = processed_dataset.y_test
        y_test = processed_dataset.X_test
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
            "cv_splits": cv_splits,
            "grid_searched_param": tune_param_name,
        }

        (
            fitted_estimator,
            loss_history,
            metrics,
            search_summary,
        ) = CrossValidationTrainer().search(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            param_name=tune_param_name,
            param_range=tune_param_range,
            validation_metrics=validation_metrics,
            parameters=parameters,
            random_state=random_state,
            cv=cv_splits,
            epochs=epochs,
            learning_curve_steps=learning_curve_steps,
        )
        self._logger.log_info("Grid search workflow completed.")

        artifact = ModelArtifact.create(
            parameters={**parameters, "grid_search_summary": search_summary},
            estimator=fitted_estimator,
            metrics=metrics,
            loss_history=loss_history,
        )

        self._model_repository.save(artifact)
