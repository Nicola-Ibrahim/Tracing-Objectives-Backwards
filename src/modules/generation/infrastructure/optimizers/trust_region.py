import numpy as np
from scipy.optimize import Bounds, minimize

from ....modeling.domain.interfaces.base_estimator import BaseEstimator


class TrustRegionOptimizer:
    """
    Infrastructure service for surrogate-assisted trust-region optimization.
    Used as a fallback for incoherent regions.
    """

    @staticmethod
    def optimize(
        surrogate: BaseEstimator,
        base_anchor: np.ndarray,
        target_objective: np.ndarray,
        trust_radius: float,
        n_candidates: int,
    ) -> np.ndarray:
        """
        Executes gradient-based optimization to tweak the base anchor toward the target objective,
        constrained strictly within a trust region.

        Args:
            surrogate: Pre-trained forward surrogate estimator.
            base_anchor: (D,) Normalized decision configuration to start from.
            target_objective: (1, 2) The exact target objective.
            trust_radius: Maximum deviation allowed from base_anchor in normalized space (0 to 1).
            n_candidates: Number of variations to return.

        Returns:
            (n_candidates, D) array of optimized candidates.
        """
        target = np.asarray(target_objective).flatten()

        # Define bounds for the trust region, keeping within [0, 1] absolute bounds
        lower_bound = np.maximum(0.0, base_anchor - trust_radius)
        upper_bound = np.minimum(1.0, base_anchor + trust_radius)
        bounds = Bounds(lower_bound, upper_bound)

        def objective_function(x):
            x_arr = x.reshape(1, -1)
            pred = surrogate.predict(x_arr)[0]
            # Euclidean distance to target
            return np.linalg.norm(pred - target)

        results = []
        rng = np.random.default_rng()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(n_candidates):
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
                results.append(res.x)

        return np.vstack(results)
