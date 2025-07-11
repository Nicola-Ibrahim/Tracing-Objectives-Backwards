import numpy as np
from scipy.spatial import ConvexHull

from .base import FeasibilityScoringStrategy


class ConvexHullScoreStrategy(FeasibilityScoringStrategy):
    """
    Strategy based on the distance to the convex hull of the Pareto front.
    Best for low-dimensional spaces due to computational complexity.
    """

    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Delta must be positive.")
        self._delta = delta

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        hull = ConvexHull(pareto_points)
        distances = [np.linalg.norm(target - pareto_points[v]) for v in hull.vertices]
        min_dist = min(distances)
        return float(1 / (1 + min_dist))  # sigmoid-like decay
