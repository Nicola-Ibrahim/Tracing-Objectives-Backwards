"""Infrastructure min-distance scoring strategy."""

import numpy as np

from ....domain.assurance.feasibility.interfaces.scoring import (
    BaseFeasibilityScoringStrategy,
)


class MinDistanceScoreStrategy(BaseFeasibilityScoringStrategy):
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("delta must be positive")
        self._delta = delta

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        distances = np.linalg.norm(pareto_points - target, axis=1)
        min_distance = float(np.min(distances))
        return max(0.0, 1.0 - min_distance / self._delta)


__all__ = ["MinDistanceScoreStrategy"]
