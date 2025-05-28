import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .base import BaseInterpolator


class LinearInterpolator(BaseInterpolator):
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

        self.inter = None

    def fit(self) -> None:
        self.inter = interp1d(
            self.decision_vectors[:, 0],
            self.decision_vectors[:, 1],
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(self.decision_vectors[0], self.decision_vectors[-1]),
        )

    def interpolate(self, alpha: float) -> NDArray:
        """Get interpolated solution for alpha ∈ [min_alpha, max_alpha]"""
        return self.inter(alpha)
