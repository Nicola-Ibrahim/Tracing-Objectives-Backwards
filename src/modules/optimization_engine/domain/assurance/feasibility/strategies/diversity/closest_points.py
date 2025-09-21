from __future__ import annotations

import numpy as np

from .base import BaseDiversityStrategy


class ClosestPointsDiversityStrategy(BaseDiversityStrategy):
    def select_diverse_points(
        self,
        *,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.size == 0 or num_suggestions <= 0:
            return np.empty((0, pareto_front_normalized.shape[1] if pareto_front_normalized.ndim else 0))
        distances = np.linalg.norm(pareto_front_normalized - target_normalized, axis=1)
        idx = np.argsort(distances)[:num_suggestions]
        return pareto_front_normalized[idx]


__all__ = ["ClosestPointsDiversityStrategy"]
