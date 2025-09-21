"""Infrastructure scoring using a convex hull containment check."""

import numpy as np
from scipy.spatial import ConvexHull

from ....domain.assurance.interfaces.scoring import FeasibilityScoringStrategy


class ConvexHullScoreStrategy(FeasibilityScoringStrategy):
    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        hull = ConvexHull(pareto_points)
        inequalities = hull.equations
        lhs = inequalities[:, :-1]
        rhs = -inequalities[:, -1]
        satisfied = np.all(lhs @ target.T <= rhs[:, None] + 1e-9)
        return 1.0 if satisfied else 0.0


__all__ = ["ConvexHullScoreStrategy"]
