from dataclasses import dataclass
from typing import Any

import numpy as np

from ....modeling.domain.entities.trained_pipeline import TrainedPipeline
from ....modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
    ProbabilisticEstimator,
)


@dataclass
class SelectionResult:
    """Result of selecting the best candidate."""

    best_index: int
    best_distance: float
    best_decision: np.ndarray
    best_objective: np.ndarray
    all_distances: np.ndarray  # For visualization
    sorted_candidates: np.ndarray  # [NEW]
    sorted_objectives: np.ndarray  # [NEW]


class InverseModelsCandidatesComparator:
    """
    Compares multiple inverse estimators by observing how each generates
    candidate decisions for a target objective.

    Pipeline per estimator:
    1. Sample: Generate candidate decisions from inverse estimator
    2. Simulate: Predict objectives via forward model
    3. Validate: (Optional) Filter candidates using OOD/Conformal validators
    4. Select: Pick best candidate by distance to target
    """

    def compare(
        self,
        inverse_pipelines: dict[str, TrainedPipeline],
        forward_pipeline: TrainedPipeline,
        target_objective_norm: dict[str, np.ndarray],
        target_objective_raw: np.ndarray,
        n_samples: int,
    ) -> dict[str, Any]:
        """
        Runs the comparison pipeline for each inverse pipeline.

        Returns:
            - results_map: Per-estimator results with selection info and distances
            - generator_runs: Pre-validation runs for visualization
        """
        results_map: dict[str, dict] = {}
        generator_runs: list[dict[str, object]] = []

        for name, pipeline in inverse_pipelines.items():
            estimator = pipeline.model.fitted

            # 1. Sample (Normalized space -> Physical space)
            target_norm = target_objective_norm[name]
            candidates_norm = self._sample(
                estimator=estimator,
                target_objective_norm=target_norm,
                n_samples=n_samples,
            )

            candidates_raw = candidates_norm.copy()
            for t in reversed(pipeline.get_decisions_transforms()):
                candidates_raw = t.inverse_transform(candidates_raw)

            # 2. Simulate (Predict outcomes)
            fwd_candidates_norm = candidates_raw.copy()
            for t in forward_pipeline.get_decisions_transforms():
                fwd_candidates_norm = t.transform(fwd_candidates_norm)

            predicted_objectives_norm_fwd = forward_pipeline.model.fitted.predict(
                fwd_candidates_norm
            )

            predicted_objectives_raw = predicted_objectives_norm_fwd.copy()
            for t in reversed(forward_pipeline.get_objectives_transforms()):
                predicted_objectives_raw = t.inverse_transform(predicted_objectives_raw)

            # Keep pre-validation run for visualization
            generator_runs.append(
                {
                    "name": name,
                    "decisions": candidates_raw,
                    "predicted_objectives": predicted_objectives_raw,
                }
            )

            # 3. Normalize for fair distance calculation
            predicted_objectives_norm = predicted_objectives_raw.copy()
            for t in pipeline.get_objectives_transforms():
                predicted_objectives_norm = t.transform(predicted_objectives_norm)

            # 4. Select (Sort all candidates by distance)
            if len(candidates_raw) > 0:
                selection = self._select(
                    candidates=candidates_raw,
                    predicted_objectives_norm=predicted_objectives_norm,
                    target_objective_norm=target_norm,
                    predicted_objectives_raw=predicted_objectives_raw,
                )

                # Update generator_runs with SORTED points and best point info
                generator_runs[-1].update(
                    {
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

    def _sample(
        self,
        estimator: BaseEstimator | ProbabilisticEstimator,
        target_objective_norm: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Samples candidate decisions from the inverse estimator.
        """
        candidates = estimator.sample(
            X=target_objective_norm,
            n_samples=n_samples,
        )

        # Convert returned candidates set from 3D to 2D: (n_candidates, x_dim).
        if candidates.ndim == 3:
            candidates = candidates.reshape(-1, candidates.shape[-1])

        return candidates

    def _select(
        self,
        candidates: np.ndarray,
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
        sorted_candidates = candidates[sort_idx]
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
