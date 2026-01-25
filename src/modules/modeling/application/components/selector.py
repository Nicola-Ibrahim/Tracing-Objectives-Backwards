from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    """Result of selecting the best candidate."""

    best_index: int
    best_distance: float
    best_decision: np.ndarray
    best_objective: np.ndarray
    all_distances: np.ndarray  # For visualization


class CandidateSelector:
    """Selects the best candidate by minimum L2 distance to target."""

    def select(
        self,
        candidates: np.ndarray,
        predicted_objectives: np.ndarray,
        target_objective_raw: np.ndarray,
    ) -> SelectionResult:
        """Returns selection result with best candidate and all distances."""
        distances = np.linalg.norm(
            predicted_objectives - target_objective_raw.reshape(1, -1), axis=1
        )
        best_idx = int(np.argmin(distances))

        return SelectionResult(
            best_index=best_idx,
            best_distance=float(distances[best_idx]),
            best_decision=candidates[best_idx],
            best_objective=predicted_objectives[best_idx],
            all_distances=distances,
        )
