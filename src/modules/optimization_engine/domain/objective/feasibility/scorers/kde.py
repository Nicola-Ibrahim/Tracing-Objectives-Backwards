import numpy as np
from scipy.stats import gaussian_kde

from .base import FeasibilityScoringStrategy


class KDEScoreStrategy(FeasibilityScoringStrategy):
    def __init__(self, bandwidth: float = 0.1):
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        self._bandwidth = bandwidth
        self._kde = None
        self._baseline_stats = None

    def fit(self, pareto_points: np.ndarray):
        self._kde = gaussian_kde(pareto_points.T, bw_method=self._bandwidth)
        log_densities = np.log(self._kde(pareto_points.T))
        self._baseline_stats = {
            "mean": np.mean(log_densities),
            "std": np.std(log_densities),
        }

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        if self._kde is None:
            self.fit(pareto_points)

        log_density = np.log(self._kde(target.T))[0]
        zscore = (log_density - self._baseline_stats["mean"]) / (
            self._baseline_stats["std"] + 1e-8
        )
        return float(1 / (1 + np.exp(-zscore)))  # sigmoid(z-score)
