from dataclasses import dataclass
from typing import Any

import numpy as np

from ....modules.inverse.domain.entities.inverse_mapping_engine import (
    InverseMappingEngine,
)


@dataclass
class SelectionResult:
    """Result of selecting the best candidate."""

    best_index: int
    best_distance: float
    best_decision: list
    best_objective: list
    all_distances: list  # For visualization
    sorted_candidates: list
    sorted_objectives: list


class InverseModelsCandidatesComparator:
    """
    Compares multiple inverse engines by observing how each generates
    candidate decisions for a target objective.
    """

    @staticmethod
    def compare(
        engines: dict[str, InverseMappingEngine],
        target_objective: np.ndarray,
        n_samples: int,
    ) -> dict[str, Any]:
        """
        Runs the comparison pipeline for each inverse engine.

        Returns:
            - results_map: Per-engine results with selection info and distances
            - generator_runs: Data for visualization
        """
        results_map: dict[str, dict] = {}
        generator_runs: list[dict[str, object]] = []

        for name, engine in engines.items():
            # 1. Transform target to engine's normalized space
            target_norm = engine.transform_objective(
                target_objective.reshape(1, -1)
            ).flatten()

            # 2. Generate (Batch generates candidates and forward simulates them)
            res = engine.solver.generate(target_y=target_norm, n_samples=n_samples)

            # 3. Denormalize for physical space display
            candidates_raw_X = engine.inverse_transform_decision(res.candidates_X)
            candidates_raw_y = engine.inverse_transform_objective(res.candidates_y)

            # 4. Select (Sort all candidates by distance in normalized objective space)
            if len(candidates_raw_X) > 0:
                selection = self._select(
                    candidates_X_raw=candidates_raw_X,
                    target_y_norm=target_norm,
                    candidates_y_norm=res.candidates_y,
                    candidates_y_raw=candidates_raw_y,
                )

                generator_runs.append(
                    {
                        "name": name,
                        "decisions": selection.sorted_candidates,
                        "predicted_objectives": selection.sorted_objectives,
                        "best_index": 0,
                        "best_decision": selection.best_decision,
                        "best_objective": selection.best_objective,
                    }
                )

                results_map[name] = {
                    "decisions": selection.sorted_candidates,
                    "predicted_objectives": selection.sorted_objectives,
                    "best_index": 0,
                    "best_distance": selection.best_distance,
                    "best_decision": selection.best_decision,
                    "best_objective": selection.best_objective,
                    "all_distances": selection.all_distances,
                }

        return {
            "results_map": results_map,
            "generator_runs": generator_runs,
        }

    def _select(
        self,
        candidates_X_raw: np.ndarray,
        target_y_norm: np.ndarray,
        candidates_y_norm: np.ndarray,
        candidates_y_raw: np.ndarray,
    ) -> SelectionResult:
        """Returns selection result with candidates sorted by NORMALIZED distance."""
        distances = np.linalg.norm(
            candidates_y_norm - target_y_norm.reshape(1, -1), axis=1
        )

        # Sort all arrays by distance
        sort_idx = np.argsort(distances)
        sorted_distances = distances[sort_idx]
        sorted_candidates = candidates_X_raw[sort_idx]
        sorted_objectives = candidates_y_raw[sort_idx]

        return SelectionResult(
            best_index=0,
            best_distance=float(sorted_distances[0]),
            best_decision=sorted_candidates[0].tolist(),
            best_objective=sorted_objectives[0].tolist(),
            all_distances=sorted_distances.tolist(),
            sorted_candidates=sorted_candidates.tolist(),
            sorted_objectives=sorted_objectives.tolist(),
        )
