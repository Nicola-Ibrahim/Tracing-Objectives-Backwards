import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class LinearInterpolator:
    def __init__(self, decision_vectors: NDArray, alphas: NDArray):
        """
        Initialize interpolator with specific solutions and their alpha positions.

        Args:
            decision_vectors: Subset of Pareto set solutions (n_points, n_parameters)
            alphas: Corresponding alpha positions [0,1] (must be sorted)
        """
        # Sort inputs by alpha
        sort_idx = np.argsort(alphas)
        self.alphas = alphas[sort_idx]
        self.decision_vectors = decision_vectors[sort_idx]

        self.interpolator = interp1d(
            self.alphas,
            self.decision_vectors,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(self.decision_vectors[0], self.decision_vectors[-1]),
        )

    def __call__(self, alpha: float) -> NDArray:
        """Get interpolated solution for alpha ∈ [min_alpha, max_alpha]"""
        return self.interpolator(alpha)
