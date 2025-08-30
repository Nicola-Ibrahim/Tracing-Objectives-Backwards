import math
from turtle import clone
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    learning_curve,
    train_test_split,
)

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


class TrainerService:
    """
    TrainerService: trains estimators, computes metrics and a unified loss_history,
    and returns a ModelArtifact.
    """

    # ---------------- Helpers ----------------

    def _evaluate_point_metrics(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, float]:
        """Evaluate a dict of metrics returning scalar scores."""

        if X is None or len(X) == 0:
            return {name: float("nan") for name in metrics}

        if isinstance(estimator, ProbabilisticEstimator):
            y_pred = estimator.predict(X, mode="mean")
        else:
            y_pred = estimator.predict(X)

        results: dict[str, float] = {}
        for name, metric in metrics.items():
            try:
                results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    # ------------------ loss_history builder ------------------

    def _build_training_chunks_for_deterministic(
        self,
        estimator: DeterministicEstimator,
        X_train: np.typing.NDArray,
        y_train: np.typing.NDArray,
        *,
        X_val: np.typing.NDArray = None,
        y_val: np.typing.NDArray = None,
        X_test: np.typing.NDArray = None,
        y_test: np.typing.NDArray = None,
        metrics: dict[str, BaseValidationMetric],
        learning_curve_steps: int = 10,
        random_state: int = 0,
    ) -> dict[str, Any]:
        """
        Build canonical training_history for deterministic estimator by subsampling
        training set at a grid of fractions and retraining clones.

        Returns canonical dict:
        {
          "bin_type": "train_fraction",
          "bins": [...fractions...],
          "n_train": [...counts...],
          "train_loss": [...],
          "val_loss": [...],
          "test_loss": [...],
        }

        Notes:
         - Uses metric named "MSE" if present, otherwise the first provided metric key.
         - train_loss/val_loss/test_loss are the chosen metric values (float) or None on failure.
        """
        # decide which metric to use as "loss" for plotting (prefer MSE)
        if metrics is None or len(metrics) == 0:
            raise ValueError("metrics dict must be provided and non-empty")

        metric_names = list(metrics.keys())
        loss_metric_name = "MSE" if "MSE" in metric_names else metric_names[0]

        n_total = len(X_train)
        fractions = list(np.linspace(0.1, 1.0, learning_curve_steps))
        rng = np.random.RandomState(random_state)

        bins: list[float] = []
        n_train_list: list[int] = []
        train_loss_list: list[float] = []
        val_loss_list: list[float] = []
        test_loss_list: list[float] = []

        for frac in fractions:
            n = max(1, int(math.floor(frac * n_total)))
            bins.append(float(frac))
            n_train_list.append(int(n))

            # subsample
            idx = rng.choice(np.arange(n_total), size=n, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train[idx]

            # clone and fit
            clonned_estimator = estimator.clone()
            try:
                # deterministic estimators generally don't accept epochs/batch args
                clonned_estimator.fit(X_sub, y_sub)
            except TypeError:
                # if clone.fit expects different args, try with kwargs-less call
                clonned_estimator.fit(X_sub, y_sub)

            # evaluate chosen loss metric on train subset
            try:
                train_scores_sub = self._evaluate_point_metrics(
                    clonned_estimator, X_sub, y_sub, metrics
                )
                t_loss = float(train_scores_sub.get(loss_metric_name, np.nan))
            except Exception:
                t_loss = None
            train_loss_list.append(t_loss)

            # evaluate val (if provided) else None
            if X_val is not None and y_val is not None:
                try:
                    val_scores_sub = self._evaluate_point_metrics(
                        clonned_estimator, X_val, y_val, metrics
                    )
                    v_loss = float(val_scores_sub.get(loss_metric_name, np.nan))
                except Exception:
                    v_loss = None
            else:
                v_loss = None
            val_loss_list.append(v_loss)

            # evaluate test (if provided) else None
            if X_test is not None and y_test is not None:
                try:
                    test_scores_sub = self._evaluate_point_metrics(
                        clonned_estimator, X_test, y_test, metrics
                    )
                    ts_loss = float(test_scores_sub.get(loss_metric_name, np.nan))
                except Exception:
                    ts_loss = None
            else:
                ts_loss = None
            test_loss_list.append(ts_loss)

        return {
            "bin_type": "train_fraction",
            "bins": bins,
            "n_train": n_train_list,
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "test_loss": test_loss_list,
        }

    # ------------------ Public API ------------------

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
        learning_curve_steps: int = 20,
        batch_size: int = 64,
        epochs: int = 100,
    ) -> ModelArtifact:
        """
        Train estimator on X/y (normalized), compute metrics and a unified loss_history,
        and return a ModelArtifact.
        """
        # Basic checks
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        # 1) train/test split
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2) normalize data
        X_train = X_normalizer.fit_transform(X_train_raw)
        X_test = X_normalizer.transform(X_test_raw)
        y_train = y_normalizer.fit_transform(y_train_raw)
        y_test = y_normalizer.transform(y_test_raw)

        # 3) train the model
        if isinstance(estimator, ProbabilisticEstimator):
            estimator.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            loss_history = estimator.get_loss_history()
            # prefer estimator.get_loss_history() that returns per-epoch dict like {"epochs":[...],"train_loss":[...],"val_loss":[...]}
            if hasattr(estimator, "get_loss_history"):
                try:
                    training_history = estimator.get_loss_history()
                    # normalize to epoch-format: if the estimator produced per-epoch train/val losses we save them as-is
                    training_history = {
                        "bin_type": "epoch",
                        "bins": list(
                            training_history.get(
                                "epochs",
                                list(
                                    range(len(training_history.get("train_loss", [])))
                                ),
                            )
                        ),
                        "n_train": [len(X_train)]
                        * len(training_history.get("train_loss", [])),
                        "train_loss": [
                            float(x) for x in training_history.get("train_loss", [])
                        ],
                        "val_loss": [
                            float(x) if x is not None else None
                            for x in training_history.get("val_loss", [])
                        ],
                        "test_loss": [None]
                        * len(training_history.get("train_loss", [])),
                    }
                except Exception:
                    training_history = None

        elif isinstance(estimator, DeterministicEstimator):
            # deterministic estimators: fit on full train set, then create training chunks via clones
            estimator.fit(X_train, y_train)
            # build training_history by subsampling + retraining clones
            training_history = self._build_training_chunks_for_deterministic(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_val=None,
                y_val=None,
                X_test=X_test,
                y_test=y_test,
                metrics=metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
            )
        else:
            # unknown type: try generic fit and no detailed history
            estimator.fit(X_train, y_train)

        # 4) Compute final scores
        train_scores = self._evaluate_point_metrics(
            estimator, X_train, y_train, metrics
        )
        test_scores = self._evaluate_point_metrics(estimator, X_test, y_test, metrics)

        # 5) Build artifact
        artifact = ModelArtifact.create(
            parameters=parameters,
            estimator=estimator,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
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
        and return ModelArtifact with cv_scores and canonical loss_history for final model.
        """

        # 1) initial holdout split + normalization
        X_train_full_raw, X_test_raw, y_train_full_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_full = X_normalizer.fit_transform(X_train_full_raw)
        X_test = X_normalizer.transform(X_test_raw)
        y_train_full = y_normalizer.fit_transform(y_train_full_raw)
        y_test = y_normalizer.transform(y_test_raw)

        # 2) k-fold CV on training portion
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
            X_train = X_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_train = y_train_full[train_idx]
            y_val = y_train_full[val_idx]

            cloned_estimator = estimator.clone()
            cloned_estimator.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

            fold_scores = self._evaluate_point_metrics(
                cloned_estimator, X_val, y_val, validation_metrics
            )
            for mname, score in fold_scores.items():
                cv_scores[mname].append(score)
                if verbose:
                    print(f"  {mname}: {score:.4f}")

        # 3) retrain final estimator on full training set
        final_estimator = estimator.clone()
        if isinstance(final_estimator, ProbabilisticEstimator):
            final_estimator.fit(
                X_train_full, y_train_full, batch_size=batch_size, epochs=epochs
            )

            # evaluate final model
            test_scores = self._evaluate_point_metrics(
                final_estimator, X_test, y_test, validation_metrics
            )
            train_scores = self._evaluate_point_metrics(
                final_estimator, X_train_full, y_train_full, validation_metrics
            )

            # get per-epoch history for final model if available (probabilistic) else None (use train_size)
            loss_history = final_estimator.get_loss_history()

        elif isinstance(final_estimator, DeterministicEstimator):
            final_estimator.fit(X_train_full, y_train_full)

            loss_history = self._build_training_chunks_for_deterministic(
                estimator=final_estimator,
                X_train=X_train_full,
                y_train=y_train_full,
                X_val=None,
                y_val=None,
                X_test=X_test,
                y_test=y_test,
                metrics=validation_metrics,
                learning_curve_steps=learning_curve_steps,
                random_state=random_state,
            )

        artifact = ModelArtifact(
            parameters=parameters,
            estimator=final_estimator,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            metadata={"loss_history": loss_history},
        )

        return artifact
