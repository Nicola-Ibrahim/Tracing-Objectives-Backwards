import numpy as np
from scipy.optimize import Bounds, minimize

from ......modeling.domain.interfaces.base_estimator import BaseEstimator


class IncoherentSampling:
    """
    Samples candidates for the incoherent pathway (within mesh bounds).
    Uses gradient-based optimization to find a local minimum and then
    generates a directional cloud bounded between the anchor and the optimum.
    """

    def __init__(
        self,
        forward_estimator: BaseEstimator,
        target_y: np.ndarray,
        trust_radius: float,
    ):
        self._forward_estimator = forward_estimator
        self._target_y = target_y
        self._trust_radius = trust_radius

    def sample(
        self,
        vertices_X: np.ndarray,
        weights: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        # 1. Setup base anchor (highest weight vertex)
        best_anchor_idx = int(np.argmax(weights))
        base_anchor = vertices_X[best_anchor_idx]
        target = np.asarray(self._target_y).flatten()

        # 2. Define trust-region bounds
        lower_bound = base_anchor - self._trust_radius
        upper_bound = base_anchor + self._trust_radius
        bounds = Bounds(lower_bound, upper_bound)

        def objective_function(x):
            x_arr = x.reshape(1, -1)
            pred = self._forward_estimator.predict(x_arr)[0]
            return np.linalg.norm(pred - target)

        # 3. Optimize to find the best local point
        res = minimize(
            objective_function,
            base_anchor,
            method="trust-constr",
            bounds=bounds,
            options={"maxiter": 50, "disp": False},
        )
        optimal_x = np.clip(res.x, lower_bound, upper_bound)

        # 4. Generate directional cloud (interpolation)
        # Axis: base_anchor -> optimal_x
        direction = optimal_x - base_anchor
        dist = np.linalg.norm(direction)

        results = []
        rng = np.random.default_rng()

        # We always include the optimal_x as the first candidate
        results.append(optimal_x)

        if n_samples > 1:
            # Beta distribution biased toward optimal_x (t=1.0)
            t_values = rng.beta(3.0, 1.5, size=n_samples - 1)

            # Lateral noise scale (5% of the distance)
            lateral_scale = dist * 0.05 if dist > 0 else 0.01

            for t in t_values:
                # Primary axis interpolation
                candidate = base_anchor + t * direction

                # Add small lateral noise for cloud thickness
                if dist > 0:
                    noise = rng.normal(0, lateral_scale, size=base_anchor.shape)
                    candidate += noise

                # Ensure we stay within trust region
                results.append(np.clip(candidate, lower_bound, upper_bound))

        return np.vstack(results)
