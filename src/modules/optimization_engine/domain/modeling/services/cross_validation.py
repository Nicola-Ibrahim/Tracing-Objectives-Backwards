from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold

from ..interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ..interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ..value_objects.loss_history import LossHistory
from ..value_objects.metrics import Metrics
from .deterministic import DeterministicModelTrainer
from .probabilistic import ProbabilisticModelTrainer
from .utils import evaluate_metrics


class CrossValidationTrainer:
    def __init__(self) -> None:
        self._det_trainer = DeterministicModelTrainer()
        self._prob_trainer = ProbabilisticModelTrainer()

    def validate(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        validation_metrics: dict[str, BaseValidationMetric],
        *,
        random_state: int = 0,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        learning_curve_steps: int = 50,
    ) -> tuple[BaseEstimator, LossHistory, Metrics]:
        """
        Split + normalize, run k-fold CV, fit final estimator on full train portion,
        compute train/test metrics and return tuple (includes normalized arrays).
        """

        # 1) CV
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores_by_fold: list[dict[str, float]] = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            estimator_clonned = estimator.clone()
            if isinstance(estimator_clonned, ProbabilisticEstimator):
                fold_fitted_estimator, fold_loss_history, fold_metrics = (
                    self._prob_trainer.train(
                        estimator_clonned,
                        X_train=X_tr,
                        y_train=y_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                )
            else:
                fold_fitted_estimator, fold_loss_history, fold_metrics = (
                    self._det_trainer.train(
                        estimator_clonned,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_test=X_val,
                        y_test=y_val,
                        validation_metrics=validation_metrics,
                        learning_curve_steps=min(learning_curve_steps, 20),
                        random_state=random_state,
                    )
                )

            fold_scores = evaluate_metrics(
                fold_fitted_estimator, X_val, y_val, validation_metrics
            )
            cv_scores_by_fold.append(fold_scores)

        # 2) final fit on full training portion via trainers (preserve internals)
        final = estimator.clone()
        if isinstance(final, ProbabilisticEstimator):
            fitted_estimator, loss_history, metrics = self._prob_trainer.train(
                final,
                X_train=X_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=batch_size,
            )
        else:
            fitted_estimator, loss_history, metrics = self._det_trainer.train(
                final,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                validation_metrics=validation_metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
            )

        # 3) compute train/test point metrics
        train_mertics = evaluate_metrics(
            fitted_estimator, X_train, y_train, validation_metrics
        )
        test_metrics = evaluate_metrics(
            fitted_estimator, X_test, y_test, validation_metrics
        )

        metrics = Metrics(
            train=[train_mertics],
            test=[test_metrics],
            cv=cv_scores_by_fold,
        )

        return fitted_estimator, loss_history, metrics

    def search(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        param_name: str,
        param_range: Iterable[Any],
        validation_metrics: dict[str, BaseValidationMetric],
        parameters: dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 0,
        cv: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        learning_curve_steps: int = 20,
    ) -> tuple[BaseEstimator, LossHistory, Metrics, dict[str, Any]]:
        """
        Grid search over param_range. Returns (TrainingOutcome for chosen param, summary).
        Summary contains param_range and per-metric validation scores.
        """
        param_vals = list(param_range)
        valid_scores: dict[str, list[float]] = {
            n: [] for n in validation_metrics.keys()
        }

        for val in param_vals:
            estimator_clonned = estimator.clone()
            try:
                setattr(estimator_clonned, param_name, val)
            except Exception:
                pass

            res = self.validate(
                estimator=estimator_clonned,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                validation_metrics=validation_metrics,
                parameters={**parameters, param_name: val},
                test_size=test_size,
                random_state=random_state,
                n_splits=cv,
                epochs=epochs,
                batch_size=batch_size,
                learning_curve_steps=learning_curve_steps,
            )

            # aggregate primary metric (mean over folds)
            for m in validation_metrics.keys():
                fold_vals = [float(fold.get(m, float("nan"))) for fold in res.cv_scores]
                mean_val = (
                    float(np.nanmean(fold_vals)) if len(fold_vals) > 0 else float("nan")
                )
                valid_scores[m].append(mean_val)

        # pick best (same policy as before)
        primary = next(iter(validation_metrics.keys()))
        best_idx = int(np.nanargmax(np.array(valid_scores[primary])))
        best_val = param_vals[best_idx]

        best_clone = estimator.clone()

        setattr(best_clone, param_name, best_val)

        estimator, loss_history, metrics = self.validate(
            best_clone,
            X_train,
            y_train,
            X_test,
            y_test,
            validation_metrics,
            parameters={**parameters, param_name: best_val},
            test_size=test_size,
            random_state=random_state,
            n_splits=cv,
            epochs=epochs,
            batch_size=batch_size,
            learning_curve_steps=learning_curve_steps,
        )

        summary = {
            "param_name": param_name,
            "param_range": param_vals,
            "valid_scores": valid_scores,
            "chosen_index": best_idx,
            "chosen_value": best_val,
        }
        return estimator, loss_history, metrics, summary
