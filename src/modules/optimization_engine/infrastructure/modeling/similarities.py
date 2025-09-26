import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from ...domain.modeling.interfaces.base_similarity import BaseSimilarityCalculator


class CosineSimilarityCalculator(BaseSimilarityCalculator):
    """
    Concrete implementation of BaseSimilarityCalculator using cosine similarity.
    This class leverages sklearn's cosine_similarity for efficient computation.
    """

    def calculate_similarity(
        self,
        array: NDArray[np.floating],
        vector: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute cosine similarity between each row of `array` and the 1D `vector`.
        This function uses sklearn's cosine similarity, which internally normalizes the inputs.

        Args:
            array: A 2D NumPy array with shape (n_samples, n_features).
            vector: A 1D NumPy array with shape (n_features,).

        Returns:
            A 1D NumPy array of similarities with shape (n_samples,).

        Raises:
            ValueError: If there's a dimension mismatch between `array` and `vector`.
        """
        # Check input dimensions
        n_samples, n_features = array.shape
        if vector.shape[0] != n_features:
            raise ValueError(
                f"Dimension mismatch: array has {n_features} features, "
                f"but vector has length {vector.shape[0]}"
            )

        # Ensure vector is 2D for sklearn's cosine_similarity
        # sklearn will L2-normalize for you, no need to normalize beforehand.
        sims = sk_cosine_similarity(array, vector.reshape(1, -1))
        return sims.ravel()
