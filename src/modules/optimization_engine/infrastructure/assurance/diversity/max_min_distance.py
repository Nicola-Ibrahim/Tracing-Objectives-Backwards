"""Infrastructure diversity strategy using scipy distance computations."""

import numpy as np
from scipy.spatial.distance import cdist

from ....domain.assurance.interfaces.diversity import DiversityStrategy


class MaxMinDistanceDiversityStrategy(DiversityStrategy):
    def select_diverse_points(
        self,
        *,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray:
        if pareto_front_normalized.size == 0 or num_suggestions <= 0:
            return np.empty((0, pareto_front_normalized.shape[1]))

        if pareto_front_normalized.shape[0] <= num_suggestions:
            return pareto_front_normalized

        selected = []
        distances = np.linalg.norm(pareto_front_normalized - target_normalized, axis=1)
        first_idx = int(np.argmin(distances))
        selected.append(pareto_front_normalized[first_idx])

        remaining = np.delete(np.arange(pareto_front_normalized.shape[0]), first_idx)

        while len(selected) < num_suggestions and remaining.size:
            dist_matrix = cdist(pareto_front_normalized[remaining], np.array(selected))
            min_dist = np.min(dist_matrix, axis=1)
            candidate_idx = int(np.argmax(min_dist))
            selected_idx = remaining[candidate_idx]
            selected.append(pareto_front_normalized[selected_idx])
            remaining = np.delete(remaining, candidate_idx)

        return np.array(selected)


__all__ = ["MaxMinDistanceDiversityStrategy"]
