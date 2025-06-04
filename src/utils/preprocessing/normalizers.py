import numpy as np
from numpy.typing import NDArray


def normalize_to_hypercube(
    data: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] = None
) -> np.ndarray:
    """
    Normalize data to unit hypercube [0, 1]^n for fair comparison.

    Args:
        data: Data array (n_samples, n_features)
        bounds: Optional (min_values, max_values) for manual normalization

    Returns:
        Normalized data
    """
    if bounds is None:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
    else:
        min_vals, max_vals = bounds

    # Avoid division by zero for constant columns
    ranges = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)

    # Normalize: (value - min) / range
    return (data - min_vals) / ranges


def normalize_to_unit_vector(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert weights to unit vector (L2 normalization)"""
    l2_norm = np.linalg.norm(weights)
    if l2_norm < 1e-10:  # Handle zero vector case
        return np.zeros_like(weights)
    return weights / l2_norm
