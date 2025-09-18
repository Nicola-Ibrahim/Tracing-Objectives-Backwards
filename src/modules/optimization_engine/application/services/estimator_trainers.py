import math
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold

from ...domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from ...domain.modeling.interfaces.base_normalizer import BaseNormalizer
from ...domain.modeling.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.modeling.value_objects.loss_history import LossHistory
from ...domain.modeling.value_objects.metrics import Metrics
from .utils import apply_param, evaluate_metrics


class DeterministicModelTrainer:
    """
    Train deterministic estimators and return TrainingOutcome.
    Internal processes preserved; method names clarified.
    """

    def compute_learning_curve(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        validation_metrics: dict[str, BaseValidationMetric],
        learning_curve_steps: int = 50,
        random_state: int = 0,
    ) -> tuple[BaseEstimator, LossHistory]:
        """
        (helper) Build the learning-curve style LossHistory by subsampling training set.
        Returns (possibly fitted) estimator and LossHistory.
        """

        # Validate estimator type & inputs
        if validation_metrics is None or len(validation_metrics) == 0:
            raise ValueError("metrics dict must be provided and non-empty")

        # 1) Setup sampling schedule
        n_total = len(X_train)
        fractions = list(np.linspace(0.1, 1.0, learning_curve_steps))
        rng = np.random.RandomState(random_state)

        bins: list[float] = []
        n_train_list: list[int] = []
        train_loss_list: list[float] = []
        val_loss_list: list[float] = []

        # 2) For each fraction: sample, train a clone, evaluate chosen metric
        first_metric_name = next(iter(validation_metrics.keys()))
        for frac in fractions:
            # 2.a) Determine sample size for this fraction
            n = max(1, int(math.floor(frac * n_total)))
            bins.append(float(frac))
            n_train_list.append(n)

            # 2.b) Subsample without replacement and train a clone
            train_idx = rng.choice(np.arange(n_total), size=n, replace=False)
            X_sub = X_train[train_idx]
            y_sub = y_train[train_idx]

            cloned = estimator.clone()
            cloned.fit(X_sub, y_sub)

            # 2.c) Evaluate metric on the training subsample (first metric used as primary loss)
            train_mertics = evaluate_metrics(cloned, X_sub, y_sub, validation_metrics)
            train_loss_list.append(
                float(train_mertics.get(first_metric_name, float("nan")))
            )

            # 2.d) Evaluate the same metric on the remaining training data as validation
            val_mask = np.ones(n_total, dtype=bool)
            val_mask[train_idx] = False
            val_idx = np.nonzero(val_mask)[0]
            if val_idx.size > 0:
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                val_scores = evaluate_metrics(cloned, X_val, y_val, validation_metrics)
                val_loss_list.append(
                    float(val_scores.get(first_metric_name, float("nan")))
                )
            else:
                val_loss_list.append(float("nan"))

        # 3) Build LossHistory object to return
        loss_history = LossHistory(
            bin_type="train_fraction",
            bins=bins,
            n_train=n_train_list,
            train_loss=train_loss_list,
            val_loss=val_loss_list,
            test_loss=[],
        )

        # 4) Fit the original estimator on the full training data so it is ready to use
        estimator.fit(X_train, y_train)

        return estimator, loss_history

    def train(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        *,
        learning_curve_steps: int = 50,
        random_state: int = 0,
        validation_metrics: dict[str, BaseValidationMetric],
    ) -> tuple[BaseEstimator, LossHistory, Metrics]:
        """
        Public entry to train a deterministic estimator on raw (X, y).
        Process explained with numbered comments:
          1) Split raw data into train/test
          2) Normalize train and test using provided normalizers
          3) If metrics provided, compute learning curve (subsampling) and fit final estimator
          4) Compute point metrics (train/test)
          5) Return tuple containing estimator, LossHistory and normalized arrays
        """

        # 1) compute learning curve & ensure estimator is fitted
        estimator, loss_history = self.compute_learning_curve(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            validation_metrics=validation_metrics,
            learning_curve_steps=learning_curve_steps,
            random_state=random_state,
        )

        # 2) compute point-wise train/test metrics using normalized arrays
        train_mertics = evaluate_metrics(
            estimator, X_train, y_train, validation_metrics
        )
        test_metrics = evaluate_metrics(estimator, X_test, y_test, validation_metrics)

        metrics = Metrics(
            train=[train_mertics],
            test=[test_metrics],
            cv=[],
        )

        return estimator, loss_history, metrics


class ProbabilisticModelTrainer:
    """
    Train probabilistic estimators and return TrainingOutcome.
    """

    def compute_epoch_history(
        self,
        estimator: ProbabilisticEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> tuple[BaseEstimator, LossHistory]:
        """
        (helper) Fit estimator and extract per-epoch loss history if provided by estimator.
        """

        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("EpochsCurve is for ProbabilisticEstimator only")

        # 1) Fit estimator for given number of epochs (estimator should internally record history)
        estimator.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # 2) Try to read history from common accessor names
        history = estimator.get_loss_history()

        # 3) Build LossHistory from recorded history
        bins = history.get("epochs", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        loss_history = LossHistory(
            bin_type="epoch",
            bins=bins,
            n_train=[len(X_train)] * len(bins),
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=[],
        )

        return estimator, loss_history

    def train(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> tuple[BaseEstimator, LossHistory, Metrics]:
        """
        Public entry to train a probabilistic estimator on raw (X, y).
        Step-by-step:
          1) Split raw data into train/test
          2) Normalize train and test using provided normalizers
          3) Fit the estimator for a number of epochs and extract epoch history
          4) Return tuple with fitted estimator, loss_history and normalized arrays
        """

        # 1) fit and collect epoch history
        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("ProbabilisticModelTrainer requires ProbabilisticEstimator")

        estimator, loss_history = self.compute_epoch_history(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

        # 2) produce TrainingOutcome; leave train/test point-scores None (caller can compute if desired)
        metrics = Metrics(train=[], test=[], cv=[])

        return estimator, loss_history, metrics


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

            clone = estimator.clone()
            if isinstance(clone, ProbabilisticEstimator):
                fold_fitted_estimator, fold_loss_history, fold_metrics = (
                    self._prob_trainer.train(
                        clone,
                        X_train=X_tr,
                        y_train=y_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                    )
                )
            else:
                fold_fitted_estimator, fold_loss_history, fold_metrics = (
                    self._det_trainer.train(
                        clone,
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
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
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
            c = estimator.clone()
            try:
                self._apply_param(c, param_name, val)
            except Exception:
                pass

            res = self.validate(
                c,
                X_train,
                y_train,
                X_test,
                y_test,
                validation_metrics,
                parameters={**parameters, param_name: val},
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
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

        apply_param(best_clone, param_name, best_val)

        estimator, loss_history, metrics = self.validate(
            best_clone,
            X_train,
            y_train,
            X_test,
            y_test,
            validation_metrics,
            parameters={**parameters, param_name: best_val},
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
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
