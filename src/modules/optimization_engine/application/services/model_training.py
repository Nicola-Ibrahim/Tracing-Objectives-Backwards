from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from ...domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.model_management.entities.model_artifact import ModelArtifact
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.model_management.interfaces.base_normalizer import BaseNormalizer
from .model_selection import (
    EpochsCurveService,
    LearningCurveService,
    ValidationCurveService,
)


class TrainerService:
    def __init__(self, loss_metric_name: str = "MSE") -> None:
        self.loss_metric_name = loss_metric_name
        self._epochs_svc = EpochsCurveService()
        self._learning_svc = LearningCurveService()
        self._validation_svc = ValidationCurveService()

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
            y_pred = estimator.predict(X, mode="mean")
        elif isinstance(estimator, DeterministicEstimator):
            y_pred = estimator.predict(X)

        results: dict[str, float] = {}
        for name, metric in metrics.items():
            try:
                results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    def train_and_evaluate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metrics: dict[str, BaseValidationMetric],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        parameters: dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 0,
        learning_curve_steps: int = 50,
        batch_size: int = 64,
        epochs: int = 100,
    ) -> ModelArtifact:
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

        # 2) Create the loss history
        loss_history = None

        if isinstance(estimator, ProbabilisticEstimator):
            loss_history = self._epochs_svc.run(
                estimator,
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
                plot=True,
            )
        elif isinstance(estimator, DeterministicEstimator):
            loss_history = self._learning_svc.run(
                estimator,
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test,
                metrics=metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
                plot=True,
            )

        # 3) Fit the main estimator on the full training data BEFORE any other operations.
        if isinstance(estimator, ProbabilisticEstimator):
            estimator.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        elif isinstance(estimator, DeterministicEstimator):
            estimator.fit(X_train, y_train)

        # 4) Evaluate the model
        train_scores = self._evaluate_point_metrics(
            estimator, X_train, y_train, metrics
        )
        test_scores = self._evaluate_point_metrics(estimator, X_test, y_test, metrics)

        if loss_history:
            final_test_loss = test_scores.get(self.loss_metric_name, np.nan)
            if loss_history.get("test_loss") is not None:
                loss_history["test_loss"] = [final_test_loss] * len(
                    loss_history["bins"]
                )
        else:
            loss_history = {
                "bin_type": "single_point",
                "bins": [0],
                "n_train": [len(X_train)],
                "train_loss": [train_scores.get(self.loss_metric_name, np.nan)],
                "val_loss": [None],
                "test_loss": [test_scores.get(self.loss_metric_name, np.nan)],
            }

            print(loss_history)

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=estimator,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores={},
            loss_history=loss_history,
        )

        return artifact

    def cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        validation_metrics: dict[str, BaseValidationMetric],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        parameters: dict[str, Any],
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int | None = 42,
        verbose: bool = True,
        batch_size: int = 32,
        epochs: int = 100,
    ) -> ModelArtifact:
        """
        Perform k-fold CV on training portion, retrain final estimator on full training set,
        and return ModelArtifact with cv_scores and a single-point loss_history.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_train_full_raw, X_test_raw, y_train_full_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_full = X_normalizer.fit_transform(X_train_full_raw)
        X_test = X_normalizer.transform(X_test_raw)
        y_train_full = y_normalizer.fit_transform(y_train_full_raw)
        y_test = y_normalizer.transform(y_test_raw)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores: dict[str, list[float]] = {
            name: [] for name in validation_metrics.keys()
        }

        if verbose:
            print(
                f"Starting {n_splits}-fold CV on {estimator.__class__.__name__} (training size={len(X_train_full)})"
            )

        for i, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            if verbose:
                print(f"--- Fold {i+1}/{n_splits} ---")

            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            cloned_estimator = estimator.clone()

            if isinstance(cloned_estimator, ProbabilisticEstimator):
                cloned_estimator.fit(
                    X_train, y_train, batch_size=batch_size, epochs=epochs
                )
            else:
                cloned_estimator.fit(X_train, y_train)

            fold_scores = self._evaluate_point_metrics(
                cloned_estimator, X_val, y_val, validation_metrics
            )
            for mname, score in fold_scores.items():
                cv_scores[mname].append(score)
                if verbose:
                    print(f"  {mname}: {score:.4f}")

        final_estimator = estimator.clone()
        if isinstance(final_estimator, ProbabilisticEstimator):
            final_estimator.fit(
                X_train_full, y_train_full, batch_size=batch_size, epochs=epochs
            )
        else:
            final_estimator.fit(X_train_full, y_train_full)

        train_scores = self._evaluate_point_metrics(
            final_estimator, X_train_full, y_train_full, validation_metrics
        )
        test_scores = self._evaluate_point_metrics(
            final_estimator, X_test, y_test, validation_metrics
        )

        avg_val_loss = np.mean(cv_scores.get(self.loss_metric_name, [np.nan]))

        final_loss_history = {
            "bin_type": "single_point",
            "bins": [0],
            "n_train": [len(X_train_full)],
            "train_loss": [train_scores.get(self.loss_metric_name, np.nan)],
            "val_loss": [float(avg_val_loss)],
            "test_loss": [test_scores.get(self.loss_metric_name, np.nan)],
        }

        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=final_estimator,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            loss_history=final_loss_history,
        )
        return artifact
