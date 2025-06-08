from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

SimilarityMethod = Callable[[NDArray, NDArray], NDArray]


def cosine_similarity(
    array: NDArray[np.floating],
    vector: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute cosine similarity between each row of `array` and the 1D `vector`.
    This function uses sklearn's cosine similarity, which internally normalizes the inputs.
    cosine similarity work with directional vectors, and The data should be transformed to same space before using this function.

    Args:
        array: shape (n_samples, n_features)
        vector: shape (n_features,)
        assume_normalized: has no effect, since sklearn normalizes internally

    Returns:
        similarities: shape (n_samples,)
    """

    # Check input dimensions
    n_samples, n_features = array.shape
    if vector.shape[0] != n_features:
        raise ValueError(
            f"Dimension mismatch: array has {n_features} features, "
            f"but vector has length {vector.shape[0]}"
        )

    # Compute similarity directly (sklearn will L2-normalize for you, no need to normalize beforehand)
    sims = sk_cosine_similarity(array, vector.reshape(1, -1))
    return sims.ravel()
