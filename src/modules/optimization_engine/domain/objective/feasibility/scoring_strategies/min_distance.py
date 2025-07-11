import numpy as np

from .base import FeasibilityScoringStrategy


class MinDistanceScoreStrategy(FeasibilityScoringStrategy):
    """
    Simplest strategy: converts minimum Euclidean distance to a feasibility score.
    """

    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Delta must be positive.")
        self._delta = delta

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        distances = np.linalg.norm(pareto_points - target, axis=1)
        min_distance = np.min(distances)
        score = max(0.0, 1 - min_distance / self._delta)
        return score
