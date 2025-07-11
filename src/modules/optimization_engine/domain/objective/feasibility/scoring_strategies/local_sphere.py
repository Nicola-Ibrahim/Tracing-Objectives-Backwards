import numpy as np

from .base import FeasibilityScoringStrategy


class LocalSphereScoreStrategy(FeasibilityScoringStrategy):
    """
    Strategy based on Euclidean distance to Pareto points, using a spherical neighborhood.

    For each normalized Pareto point, a hypersphere of radius `r` is drawn.
    The feasibility score of the target is:

        s(y*) = max_i [1 - ||y* - p_i||_2 / r]_+

    A score of 1 indicates that the point lies directly on a Pareto point.
    A score of 0 indicates the point is farther than radius r from all Pareto points.
    """

    def __init__(self, radius: float = 1.0, agg: str = "max"):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        if agg not in ("max", "mean"):
            raise ValueError("Aggregation must be one of: 'max', 'mean'")
        self._radius = radius
        self._agg = agg

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        target = target.reshape(1, -1) if target.ndim == 1 else target
        distances = np.linalg.norm(pareto_points - target, axis=1)
        scores = np.clip(1 - distances / self._radius, 0.0, 1.0)

        if self._agg == "max":
            return float(np.max(scores))
        elif self._agg == "mean":
            return float(np.mean(scores))
