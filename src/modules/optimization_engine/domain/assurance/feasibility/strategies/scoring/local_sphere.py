from __future__ import annotations

import numpy as np

from .base import FeasibilityScoringStrategy


class LocalSphereScoreStrategy(FeasibilityScoringStrategy):
    def __init__(self, radius: float = 0.1):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._radius = radius

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        distances = np.linalg.norm(pareto_points - target, axis=1)
        within = distances <= self._radius
        return float(np.sum(within) / max(1, within.size))


__all__ = ["LocalSphereScoreStrategy"]
