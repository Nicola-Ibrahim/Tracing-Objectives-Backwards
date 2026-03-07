import numpy as np


class DirichletSampling:
    """
    Samples barycentric weights via Dirichlet distribution to generate diverse,
    physically valid hybrid designs within a coherent simplex.

    Mathematical Note on the Dirichlet Distribution:
    The shape of the Dirichlet distribution is controlled by the alpha vector (\alpha).
    - If \alpha > 1: The distribution acts like a magnet, concentrating samples
      around the mean (the user's target).
    - If \alpha < 1: The distribution suffers from the "Corner Trap" (Boundary Concentration),
      repelling samples away from the center and forcing them to collapse exactly
      onto the extreme vertices of the simplex.
    """

    def __init__(self, concentration_factor: float = 100.0):
        """
        Initializes the DirichletSampling strategy.

        Args:
            concentration_factor: The multiplier used to scale the barycentric weights into
                                  the Dirichlet alpha parameters.

                                  CRITICAL: Because barycentric weights are fractions that
                                  sum to 1.0 (e.g., [0.6, 0.3, 0.1]), a large concentration
                                  factor (e.g., 50.0 to 100.0) is mathematically required
                                  to ensure all alpha values remain > 1.0.
                                  If this factor is too small (e.g., 1.0 or 0.01), alpha
                                  drops below 1.0, triggering the "Corner Trap" where the
                                  algorithm fails to interpolate and simply returns exact
                                  copies of the historical data points.
        """
        self._concentration_factor = concentration_factor

    def sample(
        self, vertices_X: np.ndarray, weights: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Samples a cloud of new candidates distributed tightly around the target.
        Calculates candidates based on barycentric weights and Dirichlet sampling.

        Args:
            vertices_X: The (3, D) decision-space configurations of the simplex vertices.
            weights: The barycentric weights representing the exact target location.
            n_samples: The number of diverse hybrid candidates to generate.

        Returns:
            (n_samples, D) array of sampled decision-space candidates.
        """
        weights = np.asarray(weights).flatten()

        # Convert fractional weights to Dirichlet alpha parameters
        alpha = weights * self._concentration_factor

        # Fallback safeguard: Dirichlet strictly requires alpha > 0.
        # We add a tiny epsilon to handle cases where a weight is exactly 0.0
        # (e.g., target is directly on an edge of the triangle).
        alpha = np.maximum(alpha, 1e-6)

        rng = np.random.default_rng()
        weight_samples = rng.dirichlet(alpha, size=n_samples)

        # Map the newly sampled probabilistic weights back to the physical Decision Space (X)
        return np.dot(weight_samples, vertices_X)
