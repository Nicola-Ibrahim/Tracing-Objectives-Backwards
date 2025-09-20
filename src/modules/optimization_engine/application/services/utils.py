from typing import Any

import numpy as np

from ...domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)
from ...domain.modeling.interfaces.base_normalizer import BaseNormalizer
from ...domain.modeling.interfaces.base_validation_metric import (
    BaseValidationMetric,
)
from .data_preparer import DataPreparer


def split_and_normalize(
    X: np.ndarray,
    y: np.ndarray,
    X_normalizer: BaseNormalizer,
    y_normalizer: BaseNormalizer,
    test_size: float,
    random_state: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, BaseNormalizer, BaseNormalizer
]:
    """
    1) Delegates to DataPreparer for a single split (keeps your splitting behavior).
    2) Applies the normalizers (fit_transform on train, transform on test).
    Returns: X_train, X_test, y_train, y_test (normalized).
    """
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = DataPreparer.single_split(
        X=X, y=y, test_size=test_size, random_state=random_state
    )

    X_train = X_normalizer.fit_transform(X_train_raw)
    X_test = X_normalizer.transform(X_test_raw)
    y_train = y_normalizer.fit_transform(y_train_raw)
    y_test = y_normalizer.transform(y_test_raw)

    return X_train, X_test, y_train, y_test, X_normalizer, y_normalizer


def safe_predict(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray | None:
    """
    Defensive predict wrapper:
      - Probabilistic estimators default to posterior mean via ``predict_mean``.
      - Deterministic estimators call ``predict`` directly.
      - Return None if predict fails.
    """
    if isinstance(estimator, ProbabilisticEstimator):
        estimator.predict_mean(X)

    return estimator.predict(X)


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

    y_pred = safe_predict(estimator, X)
    if y_pred is None:
        return {name: float("nan") for name in metrics.keys()}

    results: dict[str, float] = {}
    for name, metric in metrics.items():
        try:
            results[name] = float(metric.calculate(y_true=y, y_pred=y_pred))
        except Exception:
            results[name] = float("nan")
    return results
