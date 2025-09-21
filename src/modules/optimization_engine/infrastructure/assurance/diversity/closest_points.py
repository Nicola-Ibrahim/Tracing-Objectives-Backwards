"""Infrastructure diversity strategy selecting nearest Pareto points."""

import numpy as np

from ....domain.assurance.feasibility.interfaces.diversity import DiversityStrategy


class ClosestPointsDiversityStrategy(DiversityStrategy):
    def select_diverse_points(
        self,
        *,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.size == 0 or num_suggestions <= 0:
            return np.empty((0, pareto_front_normalized.shape[1]))
        distances = np.linalg.norm(pareto_front_normalized - target_normalized, axis=1)
        idx = np.argsort(distances)[:num_suggestions]
        return pareto_front_normalized[idx]


__all__ = ["ClosestPointsDiversityStrategy"]
