import numpy as np


class CoherentSampling:
    """
    Samples barycentric weights via Dirichlet distribution to generate diverse,
    physically valid hybrid designs within a coherent simplex.

    Mathematical Note on the Dirichlet Distribution:
    The shape of the Dirichlet distribution is controlled by the alpha vector (\alpha).
    - If \alpha > 1: Concentrates samples around the mean (the user's target).
    - If \alpha < 1: The "Corner Trap" (Boundary Concentration), where samples
      collapse exactly onto the extreme vertices of the simplex.
    """

    def __init__(self, concentration_factor: float = 100.0):
        """
        Initializes the CoherentSampling strategy.

        Args:
            concentration_factor: Multiplier to scale barycentric weights
                                  into Dirichlet alpha parameters.

                                  CRITICAL: weights sum to 1.0 (e.g. [0.6, 0.3]).
                                  A large factor (e.g., 50.0+) is required to
                                  ensure all alpha values remain > 1.0.
                                  If too small, alpha drops below 1.0, causing
                                  the "Corner Trap" where interpolation fails
                                  and simply returns copies of history points.
        """
        self._concentration_factor = concentration_factor

    def sample(
        self, vertices_X: np.ndarray, weights: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Samples a cloud of candidates distributed tightly around the target.
        Calculates candidates based on barycentrics and Dirichlet sampling.

        Args:
            vertices_X: Decision-space configurations of the simplex vertices.
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

        # Map the probabilistic weights back to the physical Decision Space (X)
        return np.dot(weight_samples, vertices_X)
