import inspect

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, train_test_split

from ...domain.model_evaluation.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
)
from .training import TrainerService


def _clone(estimator: BaseEstimator) -> BaseEstimator:
    """
    Clones an estimator by re-instantiating it with the same __init__ parameters.
    """
    klass = estimator.__class__

    # Get the signature of __init__
    sig = inspect.signature(klass.__init__)
    bound_args = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Check if the attribute exists with the same name
        if hasattr(estimator, name):
            bound_args[name] = getattr(estimator, name)
        # Check if the attribute exists as a private variable (e.g., _num_mixtures)
        elif hasattr(estimator, f"_{name}"):
            bound_args[name] = getattr(estimator, f"_{name}")
        # If not found, fall back to the default parameter value
        else:
            bound_args[name] = param.default

    # Create a new instance with the copied parameters
    return klass(**bound_args)


def cross_validate(
    estimator: BaseEstimator,
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

    pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_scores = {scorer_name: [] for scorer_name in validation_metrics.keys()}

    if verbose:
        print(
            f"Starting {n_splits}-fold cross-validation for {estimator.__class__.__name__}..."
        )

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = pre_X_train[train_index], pre_X_train[val_index]
        y_train, y_val = pre_y_train[train_index], pre_y_train[val_index]

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

    # Use TrainerService to train on full normalized data and compute learning curve + training_history
    artifact = TrainerService().train_and_evaluate(
        mapper=_clone(estimator),  # train on a fresh instance to avoid side-effects
        X=X_norm,
        y=y_norm,
        metrics=validation_metrics,
        objectives_normalizer=objectives_norm,
        decisions_normalizer=decisions_norm,
        parameters=parameters,
        test_size=0.0,  # training on full data, no internal split needed (or set small if you want)
        random_state=random_state,
        compute_learning_curve=True,
        learning_curve_steps=10,
        kwargs={"epochs": getattr(estimator, "_epochs", 50)},
    )

    # 4) Attach cross-validation scores into the artifact (we expect artifact to be your pydantic ModelArtifact)
    # artifact.cv_scores is a dict[str, list[float]] in your ModelArtifact definition — set it directly.
    try:
        artifact.cv_scores = cv_scores
    except Exception:
        # If ModelArtifact is frozen / immutable, put cv_scores into metadata as fallback
        meta = artifact.metadata or {}
        meta.update({"cv_scores": cv_scores})
        artifact.metadata = meta

    return all_scores
