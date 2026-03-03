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
    best_decision: np.ndarray
    best_objective: np.ndarray
    all_distances: np.ndarray  # For visualization
    sorted_candidates: np.ndarray
    sorted_objectives: np.ndarray


class InverseModelsCandidatesComparator:
    """
    Compares multiple inverse engines by observing how each generates
    candidate decisions for a target objective.
    """

    def compare(
        self,
        engines: dict[str, InverseMappingEngine],
        target_objective_raw: np.ndarray,
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
                target_objective_raw.reshape(1, -1)
            ).flatten()

            # 2. Generate (Batch generates candidates and forward simulates them)
            res = engine.solver.generate(target_norm, n_samples=n_samples)

            # candidates_X (decisions) and candidates_y (objectives, simulated)
            # Solvers return normalized values if trained on normalized data.
            candidates_norm_X = res.candidates_X
            candidates_norm_y = res.candidates_y

            # 3. Denormalize for physical space display
            candidates_raw_X = engine.detransform_decision(candidates_norm_X)
            candidates_raw_y = engine.detransform_objective(candidates_norm_y)

            # 4. Select (Sort all candidates by distance in normalized objective space)
            if len(candidates_raw_X) > 0:
                selection = self._select(
                    candidates_raw=candidates_raw_X,
                    predicted_objectives_norm=candidates_norm_y,
                    target_objective_norm=target_norm,
                    predicted_objectives_raw=candidates_raw_y,
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
        candidates_raw: np.ndarray,
        predicted_objectives_norm: np.ndarray,
        target_objective_norm: np.ndarray,
        predicted_objectives_raw: np.ndarray,
    ) -> SelectionResult:
        """Returns selection result with candidates sorted by NORMALIZED distance."""
        distances = np.linalg.norm(
            predicted_objectives_norm - target_objective_norm.reshape(1, -1), axis=1
        )

        # Sort all arrays by distance
        sort_idx = np.argsort(distances)
        sorted_distances = distances[sort_idx]
        sorted_candidates = candidates_raw[sort_idx]
        sorted_objectives = predicted_objectives_raw[sort_idx]

        return SelectionResult(
            best_index=0,
            best_distance=float(sorted_distances[0]),
            best_decision=sorted_candidates[0],
            best_objective=sorted_objectives[0],
            all_distances=sorted_distances,
            sorted_candidates=sorted_candidates,
            sorted_objectives=sorted_objectives,
        )
