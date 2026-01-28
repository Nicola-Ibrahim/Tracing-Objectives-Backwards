import numpy as np


def compute_sd_scale(vector: np.ndarray) -> np.ndarray:
    """
    Standard Deviation scale estimator.
    σ_j = std(y_{tr, j})

    Args:
        vector: Array of shape (m, n)

    Returns:
        Array of shape (n,) containing the scale estimates
    """
    return np.std(vector, axis=0)


def compute_mad_scale(vector: np.ndarray) -> np.ndarray:
    """
    Median Absolute Deviation (MAD) robust scale estimator.
    σ_j = 1.4826 * median(|y_{tr,j} - median(y_{tr,j})|)

    Args:
        vector: Array of shape (m, n)

    Returns:
        Array of shape (n,) containing the scale estimates
    """
    medians = np.median(vector, axis=0)
    abs_deviations = np.abs(vector - medians)
    mad = np.median(abs_deviations, axis=0)
    return 1.4826 * mad


def compute_iqr_scale(vector: np.ndarray) -> np.ndarray:
    """
    Interquartile Range (IQR) scale estimator for skewed distributions.
    σ_j = (Q_0.75 - Q_0.25) / 1.349

    Args:
        vector: Array of shape (m, n)

    Returns:
        Array of shape (n,) containing the scale estimates
    """
    q75, q25 = np.percentile(vector, [75, 25], axis=0)
    return (q75 - q25) / 1.349
