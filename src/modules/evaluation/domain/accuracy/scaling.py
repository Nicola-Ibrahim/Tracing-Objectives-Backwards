import numpy as np


def compute_sd_scale(y_train: np.ndarray) -> np.ndarray:
    """
    Standard Deviation scale estimator.
    σ_j = std(y_{tr, j})
    """
    return np.std(y_train, axis=0)


def compute_mad_scale(y_train: np.ndarray) -> np.ndarray:
    """
    Median Absolute Deviation (MAD) robust scale estimator.
    σ_j = 1.4826 * median(|y_{tr,j} - median(y_{tr,j})|)
    """
    medians = np.median(y_train, axis=0)
    abs_deviations = np.abs(y_train - medians)
    mad = np.median(abs_deviations, axis=0)
    return 1.4826 * mad


def compute_iqr_scale(y_train: np.ndarray) -> np.ndarray:
    """
    Interquartile Range (IQR) scale estimator for skewed distributions.
    σ_j = (Q_0.75 - Q_0.25) / 1.349
    """
    q75, q25 = np.percentile(y_train, [75, 25], axis=0)
    return (q75 - q25) / 1.349
