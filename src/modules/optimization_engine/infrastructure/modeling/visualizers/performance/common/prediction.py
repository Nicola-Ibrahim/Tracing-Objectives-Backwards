from typing import Any

import numpy as np

from ......domain.modeling.interfaces.base_estimator import (
    DeterministicEstimator,
    ProbabilisticEstimator,
)


def point_predict(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Stable 'point' prediction for diagnostics/surfaces."""
    if isinstance(estimator, ProbabilisticEstimator):
        y = estimator.predict(X, mode="map")
    elif isinstance(estimator, DeterministicEstimator):
        y = estimator.predict(X, mode="standard")
    y = np.asarray(y)
    return y if y.ndim == 2 else y[:, None]


def sample_band(
    estimator: Any, X: np.ndarray, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For 1D ribbon: (center, p05, p95), each (n,). Deterministic -> same curve."""
    X = np.asarray(X)
    if isinstance(estimator, ProbabilisticEstimator):
        draws = estimator.sample(X, n_samples=max(2, int(n_samples)))
        arr = np.asarray(draws)
        if arr.ndim == 2:  # (n, D) -> (n,1,D)
            arr = arr[:, None, :]
        elif arr.ndim == 3 and arr.shape[0] == n_samples:
            arr = np.transpose(arr, (1, 0, 2))  # (S,n,D) -> (n,S,D)
        if hasattr(estimator, "predict_map"):
            center = estimator.predict_map(X)
            center = center[:, 0] if center.ndim == 2 else center
        else:
            center = arr.mean(axis=1)[:, 0]
        p05 = np.percentile(arr, 5, axis=1)[:, 0]
        p95 = np.percentile(arr, 95, axis=1)[:, 0]
        return center, p05, p95
    y = estimator.predict(X)
    y = y[:, 0] if y.ndim == 2 else y
    return y, y, y
