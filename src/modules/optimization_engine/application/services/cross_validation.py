import inspect

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold

from ...domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ...domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)


def _clone(estimator: BaseInverseDecisionMapper) -> BaseInverseDecisionMapper:
    """
    Clone an estimator by re-instantiating it with the same __init__ parameters.
    Similar to sklearn.base.clone, but lightweight and framework-agnostic.
    """
    klass = estimator.__class__

    # Get the signature of __init__ and bound args from the instance
    sig = inspect.signature(klass.__init__)
    bound_args = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(estimator, name):  # if the estimator stores the arg as attribute
            bound_args[name] = getattr(estimator, name)

    # Create a new instance with the same params
    return klass(**bound_args)


def cross_validate(
    estimator: BaseInverseDecisionMapper,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    validation_metrics: dict[str, BaseValidationMetric],
    n_splits: int = 5,
    random_state: int | None = 42,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Performs k-fold cross-validation on a given inverse decision mapper or estimator.

    Args:
        mapper: An estimator object with fit/predict methods.
        X: Independent variables (features).
        y: Dependent variables (targets).
        validation_metrics: dict mapping scorer names to scoring functions.
        n_splits: Number of folds.
        random_state: Seed for reproducibility.
        verbose: If True, prints progress/results.

    Returns:
        dict[str, list[float]]: Scores from each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_scores = {scorer_name: [] for scorer_name in validation_metrics.keys()}

    if verbose:
        print(
            f"Starting {n_splits}-fold cross-validation for {estimator.__class__.__name__}..."
        )

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Clone the estimator fresh for each fold
        cloned_estimator = _clone(estimator)

        if verbose:
            print(f"--- Fold {i+1}/{n_splits} ---")

        # Fit
        cloned_estimator.fit(X_train, y_train)

        # Predict
        if hasattr(cloned_estimator, "predict"):
            # Special handling if probabilistic
            if "mode" in inspect.signature(cloned_estimator.predict).parameters:
                y_pred = cloned_estimator.predict(X_val, mode="mean")
            else:
                y_pred = cloned_estimator.predict(X_val)
        else:
            raise TypeError("Estimator must implement a predict method.")

        # Score
        for metric_name, metric_fn in validation_metrics.items():
            score = metric_fn.calculate(y_val, y_pred)
            all_scores[metric_name].append(score)
            if verbose:
                print(f"  {metric_name}: {score:.4f}")

    if verbose:
        print("\n" + "═" * 50)
        print("Cross-Validation Results")
        print("═" * 50)
        for scorer_name, scores in all_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {scorer_name}: {mean_score:.4f} ± {std_score:.4f}")
        print("═" * 50)

    return all_scores
