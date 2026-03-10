import numpy as np
from scipy.optimize import Bounds, minimize

from ......modeling.domain.interfaces.base_estimator import BaseEstimator


class GradientDescentSampling:
    """
    Samples candidates using gradient-based optimization within a trust region.
    Implements the BaseSamplingStrategy protocol.
    """

    def __init__(
        self,
        forward_estimator: BaseEstimator,
        target_y: np.ndarray,
        trust_radius: float,
    ):
        """
        Initializes the GradientDescentSampling strategy.

        Args:
            forward_estimator: The forward estimator used to predict objective values.
            target_y: The target objective values.
            trust_radius: The trust-region radius for the gradient descent.
        """
        self._forward_estimator = forward_estimator
        self._target_y = target_y
        self._trust_radius = trust_radius

    def sample(
        self,
        vertices_X: np.ndarray,
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
        base_anchor = vertices_X[best_anchor_idx]
        target = np.asarray(self._target_y).flatten()

        # Define bounds for the trust region based on the actual data distribution
        # We allow a local search around the anchor, but don't strictly enforce [0, 1]
        # if the input data doesn't actually follow that range.
        lower_bound = base_anchor - self._trust_radius
        upper_bound = base_anchor + self._trust_radius

        bounds = Bounds(lower_bound, upper_bound)

        def objective_function(x):
            x_arr = x.reshape(1, -1)
            pred = self._forward_estimator.predict(x_arr)[0]
            # Euclidean distance to target
            return np.linalg.norm(pred - target)

        # 1. Run optimization once to find the mathematical local minimum
        # Use trust-region optimization starting from the base anchor
        res = minimize(
            objective_function,
            base_anchor,
            method="trust-constr",
            bounds=bounds,
            options={
                "maxiter": 50,
                "disp": False,
            },
        )
        
        optimal_x = np.clip(res.x, lower_bound, upper_bound)
        
        # 2. Add controlled noise to generate diverse candidates within the trust region
        # We scatter samples around the optimal point, but stay strictly within bounds.
        results = []
        rng = np.random.default_rng()
        
        # Adaptive noise scale based on the average width of the trust region
        # This ensures diversity is visible regardless of the data magnitude
        region_widths = upper_bound - lower_bound
        noise_scale = np.mean(region_widths) * 0.15 
        
        for i in range(n_samples):
            if i == 0:
                # Keep the exact optimal point as the first candidate
                results.append(optimal_x)
            else:
                # Scatter others around it
                noise = rng.normal(0, noise_scale, size=optimal_x.shape)
                candidate = np.clip(optimal_x + noise, lower_bound, upper_bound)
                results.append(candidate)

        if not results:
            return np.empty((0, vertices_X.shape[1]))

        return np.vstack(results)
