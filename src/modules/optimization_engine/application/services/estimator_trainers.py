import math
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, Field
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
from .utils import apply_param, evaluate_metrics


# -------------------------
# Domain-shaped data classes
# -------------------------
class LossHistory(BaseModel):
    """
    Canonical shape for loss history returned by trainers.
    """

    bin_type: str = ""
    bins: list[float] = Field(default_factory=list)
    n_train: list[int] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    val_loss: list[float] = Field(default_factory=list)
    test_loss: list[float] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class TrainingOutcome(BaseModel):
    """
    Result returned by a trainer.
    - estimator: kept as the actual object in Python mode;
                 serialized to a small summary in JSON mode.
    - NumPy arrays: kept as arrays in Python mode; converted to lists in JSON mode.
    """

    estimator: BaseEstimator
    loss_history: LossHistory

    train_scores: dict[str, float] = None
    test_scores: dict[str, float] = None
    cv_scores: dict[str, float] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


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
        *,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        metrics: dict[str, BaseValidationMetric],
        learning_curve_steps: int = 50,
        random_state: int = 0,
    ) -> tuple[BaseEstimator, LossHistory]:
        """
        (helper) Build the learning-curve style LossHistory by subsampling training set.
        Returns (possibly fitted) estimator and LossHistory.
        """

        # Validate estimator type & inputs
        if not isinstance(estimator, DeterministicEstimator):
            raise TypeError("LearningCurve is for DeterministicEstimator only")
        if metrics is None or len(metrics) == 0:
            raise ValueError("metrics dict must be provided and non-empty")

        # 1) Setup sampling schedule
        n_total = len(X_train)
        fractions = list(np.linspace(0.1, 1.0, learning_curve_steps))
        rng = np.random.RandomState(random_state)

        bins: list[float] = []
        n_train_list: list[int] = []
        train_loss_list: list[float] = []
        val_loss_list: list[float] = []
        test_loss_list: list[float] = []

        # 2) For each fraction: sample, train a clone, evaluate chosen metric
        first_metric_name = next(iter(metrics.keys()))
        for frac in fractions:
            # 2.a) Determine sample size for this fraction
            n = max(1, int(math.floor(frac * n_total)))
            bins.append(float(frac))
            n_train_list.append(n)

            # 2.b) Subsample without replacement and train a clone
            idx = rng.choice(np.arange(n_total), size=n, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train[idx]

            cloned = estimator.clone()
            cloned.fit(X_sub, y_sub)

            # 2.c) Evaluate metric on the training subsample (first metric used as primary loss)
            train_scores = evaluate_metrics(cloned, X_sub, y_sub, metrics)
            train_loss_list.append(
                float(train_scores.get(first_metric_name, float("nan")))
            )

            # 2.d) Evaluate the same metric on X_test (if provided)
            test_scores = evaluate_metrics(cloned, X_test, y_test, metrics)
            test_loss_list.append(
                float(test_scores.get(first_metric_name, float("nan")))
            )

            # 2.e) We leave val_loss as NaN (no separate val computed by learning curve here)
            val_loss_list.append(float("nan"))

        # 3) Final fit on the entire training portion (so estimator returned is fitted)
        estimator.fit(X=y_train, y=X_train)

        # 4) Build LossHistory object to return
        loss_history = LossHistory(
            bin_type="train_fraction",
            bins=bins,
            n_train=n_train_list,
            train_loss=train_loss_list,
            val_loss=val_loss_list,
            test_loss=test_loss_list,
        )

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
        metrics: dict[str, BaseValidationMetric],
    ) -> TrainingOutcome:
        """
        Public entry to train a deterministic estimator on raw (X, y).
        Process explained with numbered comments:
          1) Split raw data into train/test
          2) Normalize train and test using provided normalizers
          3) If metrics provided, compute learning curve (subsampling) and fit final estimator
          4) Compute point metrics (train/test)
          5) Return TrainingOutcome containing estimator, LossHistory and normalized arrays
        """

        # 1) compute learning curve & ensure estimator is fitted
        if isinstance(estimator, DeterministicEstimator):
            estimator, loss_history = self.compute_learning_curve(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                metrics=metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
            )
        else:
            # fallback: fit estimator and return single-point loss history
            estimator.fit(X=y_train, y=X_train)
            loss_history = LossHistory(
                bin_type="single_point",
                bins=[],
                n_train=[len(X_train)],
                train_loss=[],
                val_loss=[],
                test_loss=[],
            )

        # 2) compute point-wise train/test metrics using normalized arrays
        train_scores = evaluate_metrics(estimator, X_train, y_train, metrics)
        test_scores = evaluate_metrics(estimator, X_test, y_test, metrics)

        # 3) return structured outcome
        return TrainingOutcome(
            estimator=estimator,
            loss_history=loss_history,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores={},
        )


class ProbabilisticModelTrainer:
    """
    Train probabilistic estimators and return TrainingOutcome.
    """

    def compute_epoch_history(
        self,
        estimator: ProbabilisticEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        *,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> tuple[BaseEstimator, LossHistory]:
        """
        (helper) Fit estimator and extract per-epoch loss history if provided by estimator.
        """

        if not isinstance(estimator, ProbabilisticEstimator):
            raise TypeError("EpochsCurve is for ProbabilisticEstimator only")

        # 1) Fit estimator for given number of epochs (estimator should internally record history)
        estimator.fit(X=y_train, y=X_train, epochs=epochs, batch_size=batch_size)

        # 2) Try to read history from common accessor names
        history = estimator.get_loss_history()

        # 3) Defensive default if no history provided
        if not history:
            return estimator, LossHistory(
                bin_type="epoch",
                bins=[],
                n_train=[],
                train_loss=[],
                val_loss=[],
                test_loss=[],
            )

        # 4) Build LossHistory from recorded history
        bins = list(history.get("epochs", []))
        train_loss = [float(x) for x in history.get("train_loss", [])]
        val_loss = [float(x) for x in history.get("val_loss", [])]

        loss_history = LossHistory(
            bin_type="epoch",
            bins=bins,
            n_train=[len(X_train)] * len(bins),
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=[float("nan")] * len(bins),
        )

        return estimator, loss_history

    def train(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> TrainingOutcome:
        """
        Public entry to train a probabilistic estimator on raw (X, y).
        Step-by-step:
          1) Split raw data into train/test
          2) Normalize train and test using provided normalizers
          3) Fit the estimator for a number of epochs and extract epoch history
          4) Return TrainingOutcome with fitted estimator, loss_history and normalized arrays
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
        return TrainingOutcome(
            estimator=estimator,
            loss_history=loss_history,
            train_scores={},
            test_scores={},
            cv_scores={},
        )


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
        metrics: dict[str, BaseValidationMetric],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        random_state: int = 0,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        learning_curve_steps: int = 20,
    ) -> TrainingOutcome:
        """
        Split + normalize, run k-fold CV, fit final estimator on full train portion,
        compute train/test metrics and return TrainingOutcome (includes normalized arrays).
        """

        # 1) CV
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores: dict[str, list[float]] = {n: [] for n in metrics.keys()}
        fold_histories: list[dict] = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            clone = estimator.clone()
            if isinstance(clone, ProbabilisticEstimator):
                trained_fold, fold_loss, _ = self._prob_trainer.train(
                    clone,
                    X_tr,
                    y_tr,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    test_size=0.0,
                )
            else:
                trained_fold, fold_loss, _ = self._det_trainer.train(
                    clone,
                    X_tr,
                    y_tr,
                    X_test=X_val,
                    y_test=y_val,
                    metrics=metrics,
                    learning_curve_steps=min(learning_curve_steps, 20),
                    test_size=0.0,
                )

            fold_histories.append(fold_loss)
            fold_scores = evaluate_metrics(trained_fold, X_val, y_val, metrics)
            for m, s in fold_scores.items():
                cv_scores[m].append(s)

        # 2) final fit on full training portion via trainers (preserve internals)
        final = estimator.clone()
        if isinstance(final, ProbabilisticEstimator):
            final_outcome = self._prob_trainer.train(
                final,
                X_train,
                y_train,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                epochs=epochs,
                batch_size=batch_size,
                test_size=0.0,
                random_state=random_state,
            )
        else:
            final_outcome = self._det_trainer.train(
                final,
                X_train,
                y_train,
                X_normalizer=X_normalizer,
                y_normalizer=y_normalizer,
                metrics=metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
                test_size=0.0,
            )

        final_trained = final_outcome.estimator
        final_loss = final_outcome.loss_history

        # 4) compute train/test point metrics
        train_scores = evaluate_metrics(final_trained, X_train, y_train, metrics)
        test_scores = evaluate_metrics(final_trained, X_test, y_test, metrics)

        return TrainingOutcome(
            estimator=final_trained,
            loss_history=final_loss,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
        )

    def search(
        self,
        estimator: BaseEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        X_test: np.typing.NDArray,
        y_test: np.typing.NDArray,
        param_name: str,
        param_range: Iterable[Any],
        metrics: dict[str, BaseValidationMetric],
        parameters: dict[str, Any],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        test_size: float = 0.2,
        random_state: int = 0,
        cv: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        learning_curve_steps: int = 20,
    ) -> tuple[TrainingOutcome, dict[str, Any]]:
        """
        Grid search over param_range. Returns (TrainingOutcome for chosen param, summary).
        Summary contains param_range and per-metric validation scores.
        """
        param_vals = list(param_range)
        valid_scores: dict[str, list[float]] = {n: [] for n in metrics.keys()}

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
                metrics,
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
            for m in metrics.keys():
                mean_val = float(np.nanmean(res.cv_scores.get(m, [float("nan")])))
                valid_scores[m].append(mean_val)

        # pick best (same policy as before)
        primary = next(iter(metrics.keys()))
        best_idx = int(np.nanargmax(np.array(valid_scores[primary])))
        best_val = param_vals[best_idx]

        best_clone = estimator.clone()

        apply_param(best_clone, param_name, best_val)

        best_result = self.validate(
            best_clone,
            X_train,
            y_train,
            X_test,
            y_test,
            metrics,
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
        return best_result, summary
