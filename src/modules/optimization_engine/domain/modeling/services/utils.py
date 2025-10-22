import numpy as np

from ..interfaces.base_estimator import BaseEstimator
from ..interfaces.base_validation_metric import BaseValidationMetric


def evaluate_metrics(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    metrics: dict[str, BaseValidationMetric],
) -> dict[str, float]:
    """
    Nan-safe evaluation: uses safe_predict and returns a dict metric->float.
    Mirrors the behaviour you had in multiple places.
    """
    if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
        return {name: float("nan") for name in metrics.keys()}

    if isinstance(estimator, ProbabilisticEstimator):
        y_pred = estimator.predict_mean(X, n_samples=256, seed=42)
    else:
        y_pred = estimator.predict(X)

    if y_pred is None:
        return {name: float("nan") for name in metrics.keys()}

    results: dict[str, float] = {}
    for name, metric in metrics.items():
        try:
            results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
        except Exception:
            results[name] = float("nan")
    return results
