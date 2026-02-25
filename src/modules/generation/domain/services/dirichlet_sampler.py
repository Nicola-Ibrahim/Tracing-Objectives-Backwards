import numpy as np


class DirichletSampler:
    """
    Domain service for sampling barycentric weights via Dirichlet distribution.
    """

    @staticmethod
    def sample(
        barycentric_weights: np.ndarray, n_samples: int, concentration_factor: float
    ) -> np.ndarray:
        """
        Samples a cloud of new weight vectors distributed tightly around the exact target weights.

        Args:
            barycentric_weights: (V,) Exact weights for the target.
            n_samples: Number of samples to draw.
            concentration_factor: Scalar controlling the tightness of the distribution.

        Returns:
            (n_samples, V) array of sampled weights.
        """
        weights = np.asarray(barycentric_weights).flatten()

        # Dirichlet requires alpha > 0. We add a tiny epsilon to handle exactly 0 weights
        alpha = weights * concentration_factor
        alpha = np.maximum(alpha, 1e-6)

        rng = np.random.default_rng()
        samples = rng.dirichlet(alpha, size=n_samples)
        return samples
