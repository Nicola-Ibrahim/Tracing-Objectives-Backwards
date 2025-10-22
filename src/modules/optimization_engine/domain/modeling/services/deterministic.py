import math

import numpy as np

from ..interfaces.base_estimator import BaseEstimator
from ..interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ..value_objects.loss_history import LossHistory
from ..value_objects.metrics import Metrics
from .utils import evaluate_metrics


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
