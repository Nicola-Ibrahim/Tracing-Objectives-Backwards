import math
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, train_test_split

from ...domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.model_management.entities.model_artifact import (
    LearningCurveRecord,
    ModelArtifact,
    TrainingHistory,
)
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ...domain.model_management.interfaces.base_normalizer import BaseNormalizer


class TrainerService:
    def __init__(self):
        """Initializes TrainerService."""
        pass

    # ---------- Helper methods (now part of the class) ----------

    def _predict_for_metrics(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        """Return a point prediction array for point metrics."""
        if isinstance(estimator, ProbabilisticEstimator):
            # For probabilistic estimators, we get the mean prediction
            return estimator.predict(X, mode="mean")
        return estimator.predict(X)

    def _eval_using_base_metrics(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: dict[str, BaseValidationMetric],
    ) -> dict[str, float]:
        """Evaluate a dict of BaseValidationMetric objects on X/y using estimator predictions."""
        if X is None or len(X) == 0:
            return {name: float("nan") for name in metrics.keys()}
        y_pred = self._predict_for_metrics(estimator, X)
        results: dict[str, float] = {}
        for name, metric in metrics.items():
            try:
                results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    # ----------------------
    # Public methods
    # ----------------------
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
        compute_learning_curve: bool = True,
        train_sizes: list[float] | None = None,
        learning_curve_steps: int = 5,
        batch_size: int = 64,
        kwargs: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """
        Train 'estimator' on (X,y), compute metrics, learning curve and produce a ModelArtifact.
        """
        kwargs = kwargs or {}

        # 0) Quick checks
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        # 1) Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2) Normalize the data
        X_normalizer.fit_transform(X_train)
        X_normalizer.transform(X_test)
        y_normalizer.fit_transform(y_train)
        y_normalizer.transform(y_test)

        # 3) Fit estimator on full train set
        actual_kwargs = dict(kwargs)
        actual_kwargs.update(
            {"X_val": X_test, "y_val": y_test, "batch_size": batch_size}
        )
        estimator.fit(X_train, y_train, **actual_kwargs)

        # 4) Training history (optional)
        training_history = None
        if hasattr(estimator, "get_training_history"):
            th = estimator.get_training_history()
            if th:
                training_history = TrainingHistory(
                    epochs=list(th.get("epochs", [])),
                    train_loss=list(th.get("train_loss", [])),
                    val_loss=list(th.get("val_loss", [])),
                )

        # 5) Compute train/test metrics
        train_scores = self._eval_using_base_metrics(
            estimator, X_train, y_train, metrics
        )
        test_scores = self._eval_using_base_metrics(estimator, X_test, y_test, metrics)

        # 6) Compute optional learning curve
        learning_curve = []
        if compute_learning_curve:
            if train_sizes is None:
                train_sizes = list(np.linspace(0.1, 1.0, learning_curve_steps))
            lc_records = []
            rng = np.random.RandomState(random_state)
            n_total = len(X_train)

            for frac in train_sizes:
                n = max(1, int(math.floor(frac * n_total)))
                idx = rng.choice(np.arange(n_total), size=n, replace=False)
                X_sub = X_train[idx]
                y_sub = y_train[idx]

                # Create a fresh clone for retraining
                estimator_cloned = estimator.clone()

                # Fit clone on subset
                estimator_cloned.fit(X_sub, y_sub, **actual_kwargs)

                train_scores_sub = self._eval_using_base_metrics(
                    estimator_cloned, X_sub, y_sub, metrics
                )
                val_scores_sub = self._eval_using_base_metrics(
                    estimator_cloned, X_test, y_test, metrics
                )

                lc_records.append(
                    LearningCurveRecord(
                        train_fraction=float(frac),
                        n_train=int(n),
                        train_scores=train_scores_sub,
                        val_scores=val_scores_sub,
                    )
                )
            learning_curve = lc_records

        # 7) Build and return ModelArtifact
        artifact = ModelArtifact(
            parameters=parameters,
            estimator=estimator,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores={},
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
            metadata={
                "learning_curve": [rec.__dict__ for rec in (learning_curve or [])],
                "training_history": training_history.__dict__
                if training_history is not None
                else None,
            },
        )

        return artifact

    def cross_validate(
        self,
        estimator: BaseEstimator,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        validation_metrics: dict[str, BaseValidationMetric],
        *,
        X_normalizer: BaseNormalizer,
        y_normalizer: BaseNormalizer,
        parameters: dict[str, Any],
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int | None = 42,
        verbose: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """
        Performs k-fold cross-validation, then refits on the full training set
        and returns a final ModelArtifact.
        """
        kwargs = kwargs or {}

        # 1) Correctly perform the initial train/test split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 2) Normalize data once for all steps
        X_normalizer.fit_transform(X_train_full)
        X_normalizer.transform(X_test)
        y_normalizer.fit_transform(y_train_full)
        y_normalizer.transform(y_test)

        # We perform normalization on the full training set so the CV folds
        # are transformed consistently. The holdout test set is also transformed
        # using the same normalizers.
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = {scorer_name: [] for scorer_name in validation_metrics}

        if verbose:
            print(
                f"Splitting data: {len(X_train_full)} training samples / {len(X_test)} test samples."
            )
            print(
                f"Starting {n_splits}-fold cross-validation for {estimator.__class__.__name__}..."
            )

        # 3) Perform cross-validation on the TRAINING data ONLY
        for i, (train_index, val_index) in enumerate(kf.split(X_train_full)):
            X_train, X_val = X_train_full[train_index], X_train_full[val_index]
            y_train, y_val = y_train_full[train_index], y_train_full[val_index]

            # Clone the estimator fresh for each fold
            estimator_clonned = estimator.clone()

            if verbose:
                print(f"--- Fold {i+1}/{n_splits} ---")

            # Fit on the training fold
            estimator_clonned.fit(X_train, y_train, **kwargs)

            # Evaluate on the validation fold
            fold_scores = self._eval_using_base_metrics(
                estimator_clonned, X_val, y_val, validation_metrics
            )

            # Store and print scores
            for metric_name, score in fold_scores.items():
                cv_scores[metric_name].append(score)
                if verbose:
                    print(f"  {metric_name}: {score:.4f}")

        if verbose:
            print("\n" + "═" * 50)
            print("Cross-Validation Results")
            print("═" * 50)
            for scorer_name, scores in cv_scores.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  {scorer_name}: {mean_score:.4f} ± {std_score:.4f} (CV)")
            print("═" * 50)

        # 4) Re-fit the final model on the entire training set
        final_estimator_clonned = estimator.clone()
        final_estimator_clonned.fit(X_train_full, y_train_full, **kwargs)

        # 5) Evaluate the final model on the holdout test set
        test_scores = self._eval_using_base_metrics(
            final_estimator_clonned, X_test, y_test, validation_metrics
        )
        # Re-evaluate on the full training set as well for completeness
        train_scores = self._eval_using_base_metrics(
            final_estimator_clonned, X_train_full, y_train_full, validation_metrics
        )

        if verbose:
            print("\n" + "═" * 50)
            print("Final Evaluation on Holdout Test Set")
            print("═" * 50)
            for metric_name, score in test_scores.items():
                print(f"  {metric_name}: {score:.4f}")
            print("═" * 50)

        # 6) Build and return the ModelArtifact
        artifact = ModelArtifact(
            parameters=parameters,
            estimator=final_estimator_clonned,
            train_scores=train_scores,
            test_scores=test_scores,
            cv_scores=cv_scores,
            X_normalizer=X_normalizer,
            y_normalizer=y_normalizer,
        )

        return artifact
