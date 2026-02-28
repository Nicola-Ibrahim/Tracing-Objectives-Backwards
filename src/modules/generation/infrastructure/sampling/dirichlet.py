import numpy as np

from ...domain.interfaces.sampling_strategy import BaseSamplingStrategy


class DirichletSampling(BaseSamplingStrategy):
    """
    Samples barycentric weights via Dirichlet distribution.
    Implements the BaseSamplingStrategy protocol.
    """

    def __init__(self, concentration_factor: float):
        self._concentration_factor = concentration_factor

    def sample(
        self, vertices: np.ndarray, weights: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Samples a cloud of new candidates distributed tightly around the target.
        Calculates candidates based on barycentric weights and Dirichlet sampling.

        Args:
            vertices: The decision-space configurations of the vertices.
            weights: The barycentric weights of the target.
            n_samples: The number of candidates to sample.

        Returns:
            (n_samples, D) array of sampled decision-space candidates.
        """
        weights = np.asarray(weights).flatten()

        # Dirichlet requires alpha > 0. We add a tiny epsilon to handle exactly 0 weights
        alpha = weights * self._concentration_factor
        alpha = np.maximum(alpha, 1e-6)

        rng = np.random.default_rng()
        weight_samples = rng.dirichlet(alpha, size=n_samples)

        # Map weight samples to decision space
        return np.dot(weight_samples, vertices)
