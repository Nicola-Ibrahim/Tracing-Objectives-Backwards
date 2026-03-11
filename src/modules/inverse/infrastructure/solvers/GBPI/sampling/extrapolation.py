import numpy as np
from scipy.optimize import Bounds, minimize
from ......modeling.domain.interfaces.base_estimator import BaseEstimator

class ExtrapolationSampling:
    """
    Samples candidates for the extrapolation pathway (outside mesh bounds).
    Uses gradient-based optimization and generates an unbounded cloud that
    can overshoot past optimal_x toward target_y.
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
        # 1. Setup base anchor (highest weight NN)
        # For extrapolation, weights might be different but we pick the best anchor
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

        # 3. Optimize
        res = minimize(
            objective_function,
            base_anchor,
            method="trust-constr",
            bounds=bounds,
            options={"maxiter": 50, "disp": False},
        )
        optimal_x = np.clip(res.x, lower_bound, upper_bound)

        # 4. Generate unbounded cloud
        # Axis: base_anchor -> optimal_x
        direction = optimal_x - base_anchor
        dist = np.linalg.norm(direction)
        
        results = []
        rng = np.random.default_rng()
        
        # Include optimal_x
        results.append(optimal_x)
        
        if n_samples > 1:
            # Uniform(0.5, 1.5) allows overshooting past optimal_x toward target_y
            t_values = rng.uniform(0.5, 1.5, size=n_samples - 1)
            
            # Larger lateral noise scale (10% of the distance) for wider exploration
            lateral_scale = dist * 0.10 if dist > 0 else 0.02
            
            for t in t_values:
                candidate = base_anchor + t * direction
                
                # Add lateral noise
                if dist > 0:
                    noise = rng.normal(0, lateral_scale, size=base_anchor.shape)
                    candidate += noise
                
                # Clip only to trust region, not to vertex-anchor bounds
                results.append(np.clip(candidate, lower_bound, upper_bound))

        return np.vstack(results)
