from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

SimilarityMetric = Callable[[NDArray, NDArray], NDArray]


def normalize_to_unit_vector(self, weights: NDArray) -> NDArray:
    """Convert weights to unit vector (L2 normalization)"""
    l2_norm = np.linalg.norm(weights)
    if l2_norm < 1e-10:  # Handle zero vector case
        return np.zeros_like(weights)
    return weights / l2_norm


def cosine_similarity(
    array: NDArray[np.floating],
    vector: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute cosine similarity between each row of `array` and the 1D `vector`.

    Args:
        array: shape (n_samples, n_features)
        vector: shape (n_features,)
        assume_normalized: has no effect, since sklearn normalizes internally

    Returns:
        similarities: shape (n_samples,)
    """
    # Check input dimensions
    if array.ndim != 2:
        raise ValueError(f"`array` must be 2D, got shape {array.shape}")
    if vector.ndim != 1:
        raise ValueError(f"`vector` must be 1D, got shape {vector.shape}")

    n_samples, n_features = array.shape
    if vector.shape[0] != n_features:
        raise ValueError(
            f"Dimension mismatch: array has {n_features} features, "
            f"but vector has length {vector.shape[0]}"
        )

    # Compute similarity directly (sklearn will L2-normalize for you, no need to normalize beforehand)
    sims = sk_cosine_similarity(array, vector.reshape(1, -1))
    return sims.ravel()
