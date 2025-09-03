from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from ...domain.model_management.entities.model_artifact import ModelArtifact
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.model_management.interfaces.base_normalizer import BaseNormalizer
from ...domain.model_management.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...infrastructure.visualizers.learning import (
    EpochsCurve,
    LearningCurve,
    ValidationCurve,
)


class TrainerService:
    def __init__(self, loss_metric_name: str = "MSE") -> None:
        self.loss_metric_name = loss_metric_name
        self._epochs_svc = EpochsCurve()
        self._learning_svc = LearningCurve()
        self._validation_svc = ValidationCurve()

    def _evaluate_point_metrics(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, float]:
        if X is None or len(X) == 0:
            return {name: float("nan") for name in metrics}

        if isinstance(estimator, ProbabilisticEstimator):
            y_pred = estimator.predict(X)
        elif isinstance(estimator, DeterministicEstimator):
            y_pred = estimator.predict(X)

        results: dict[str, float] = {}
        for name, metric in metrics.items():
            try:
                results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    def tune_hyperparameter(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: Iterable[Any],
        metrics: dict[str, BaseValidationMetric],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        parameters: dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 0,
        cv: int = 5,
    ) -> ModelArtifact:
        """
        Trains and evaluates a model for a range of parameter values to generate a validation curve.

        Args:
            estimator: The model to be trained. It must have a `clone()` method.
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            param_name (str): The name of the parameter to vary.
            param_range (Iterable[Any]): The range of parameter values to test.
            metrics (dict[str, BaseValidationMetric]): The metrics to evaluate the model's performance.
            X_normalizer (BaseNormalizer): The normalizer for the input features.
            y_normalizer (BaseNormalizer): The normalizer for the target values.
            parameters (dict[str, Any]): A dictionary of other model parameters.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed for the random number generator.
            cv (int): The number of cross-validation splits.

        Returns:
            ModelArtifact: An artifact containing the trained model and validation curve data.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        # 1) Split and Normalize the data
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train = X_normalizer.fit_transform(X_train_raw)
        X_test = X_normalizer.transform(X_test_raw)
        y_train = y_normalizer.fit_transform(y_train_raw)
        y_test = y_normalizer.transform(y_test_raw)

        # 2) Generate validation curve data
        validation_curve_results = self._validation_svc.run(
            estimator,
            X_train,
            y_train,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            metrics=metrics,
            random_state=random_state,
        )

        # 3) Fit a final model on the full training data with a representative parameter
        # We choose the first parameter in the range for this example, but a better approach
        # would be to select the best one based on validation scores.
        final_estimator = estimator.clone()
        self._validation_svc._set_param(
            final_estimator, param_name, validation_curve_results["param_range"][0]
        )

        if isinstance(final_estimator, ProbabilisticEstimator):
            final_estimator.fit(X_train, y_train, epochs=100, batch_size=64)
        else:
            final_estimator.fit(X_train, y_train)

        # 4) Evaluate the final model on train and test sets
        train_scores = self._evaluate_point_metrics(
            final_estimator, X_train, y_train, metrics
        )
        test_scores = self._evaluate_point_metrics(
            final_estimator, X_test, y_test, metrics
        )

        # 5) Create the ModelArtifact with the validation curve results
        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=final_estimator,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores={},
            loss_history=validation_curve_results,
        )

        return artifact, X_train, y_train, X_test, y_test
