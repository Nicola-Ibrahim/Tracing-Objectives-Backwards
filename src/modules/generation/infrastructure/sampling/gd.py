import numpy as np
from scipy.optimize import Bounds, minimize

from ....modeling.domain.interfaces.base_estimator import BaseEstimator
from ...domain.interfaces.sampling_strategy import BaseSamplingStrategy


class GradientDescentSampling(BaseSamplingStrategy):
    """
    Samples candidates using gradient-based optimization within a trust region.
    Implements the BaseSamplingStrategy protocol.
    """

    def __init__(
        self,
        surrogate_estimator: BaseEstimator,
        target_norm: np.ndarray,
        trust_radius: float,
    ):
        self._surrogate_estimator = surrogate_estimator
        self._target_norm = target_norm
        self._trust_radius = trust_radius

    def sample(
        self,
        vertices: np.ndarray,
        weights: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Executes gradient-based optimization to tweak the base anchor toward the target objective,
        constrained strictly within a trust region.

        Args:
            vertices: The decision-space configurations of the vertices.
            weights: The barycentric weights of the target.
            n_samples: The number of candidates to sample.

        Returns:
            (n_samples, D) array of optimized candidates.
        """

        # Determine base anchor from weights (highest weight anchor)
        best_anchor_idx = int(np.argmax(weights))
        base_anchor = vertices[best_anchor_idx]

        target = np.asarray(self._target_norm).flatten()

        # Define bounds for the trust region, keeping within [0, 1] absolute bounds
        lower_bound = np.maximum(0.0, base_anchor - self._trust_radius)
        upper_bound = np.minimum(1.0, base_anchor + self._trust_radius)
        bounds = Bounds(lower_bound, upper_bound)

        def objective_function(x):
            x_arr = x.reshape(1, -1)
            pred = self._surrogate_estimator.predict(x_arr)[0]
            # Euclidean distance to target
            return np.linalg.norm(pred - target)

        results = []
        rng = np.random.default_rng()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(n_samples):
                # Start from random points within the strict trust region to find multiple candidates
                x0 = rng.uniform(lower_bound, upper_bound)

                res = minimize(
                    objective_function,
                    x0,
                    method="trust-constr",
                    bounds=bounds,
                    options={
                        "maxiter": 50,
                        "disp": False,
                    },  # Keep it fast for real-time
                )
                results.append(np.clip(res.x, lower_bound, upper_bound))

        return np.vstack(results)
