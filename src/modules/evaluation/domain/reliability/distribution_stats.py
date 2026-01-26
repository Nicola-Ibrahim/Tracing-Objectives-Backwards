import numpy as np


def compute_diversity(samples: np.ndarray) -> np.ndarray:
    """
    How much the suggested solutions differ from each other for each test point.
    Returns array of shape (N_test,)
    """
    # std per dimension, mean across dimensions
    return np.std(samples, axis=1).mean(axis=1)


def compute_interval_width(
    samples: np.ndarray, quantile_width: float = 0.90
) -> np.ndarray:
    """
    Width of the predictive distribution (difference between high and low quantiles).
    Returns array of shape (N_test, x_dim)
    """
    q_high = (1 + quantile_width) / 2
    q_low = (1 - quantile_width) / 2

    q95 = np.percentile(samples, q_high * 100, axis=1)
    q05 = np.percentile(samples, q_low * 100, axis=1)

    return q95 - q05
