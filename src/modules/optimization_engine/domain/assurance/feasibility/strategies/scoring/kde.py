"""Kernel density based feasibility scoring."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from .base import FeasibilityScoringStrategy


class KDEScoreStrategy(FeasibilityScoringStrategy):
    def __init__(self, bandwidth: float = 0.1):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self._bandwidth = bandwidth
        self._kde: gaussian_kde | None = None
        self._stats: dict[str, float] | None = None

    def fit(self, pareto_points: np.ndarray) -> None:
        kde = gaussian_kde(pareto_points.T, bw_method=self._bandwidth)
        log_densities = np.log(kde(pareto_points.T))
        self._kde = kde
        self._stats = {
            "mean": float(np.mean(log_densities)),
            "std": float(np.std(log_densities) + 1e-8),
        }

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        if self._kde is None:
            self.fit(pareto_points)
        assert self._kde is not None and self._stats is not None
        log_density = float(np.log(self._kde(target.T))[0])
        z = (log_density - self._stats["mean"]) / self._stats["std"]
        return float(1.0 / (1.0 + np.exp(-z)))


__all__ = ["KDEScoreStrategy"]
