import numpy as np


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a 2-D array by promoting 1-D vectors to row matrices."""

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}")
    return arr


def clip01(values: np.ndarray) -> np.ndarray:
    """Clip values to the inclusive [0, 1] range without copying when possible."""

    return np.clip(values, 0.0, 1.0, out=values)
