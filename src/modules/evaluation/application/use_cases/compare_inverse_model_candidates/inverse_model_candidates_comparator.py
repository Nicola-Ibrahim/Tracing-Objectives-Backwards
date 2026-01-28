from dataclasses import dataclass
from typing import Any

import numpy as np

from .....modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
    DeterministicEstimator,
    ProbabilisticEstimator,
)
from .....modeling.domain.interfaces.base_normalizer import BaseNormalizer


@dataclass
class SelectionResult:
    """Result of selecting the best candidate."""

    best_index: int
    best_distance: float
    best_decision: np.ndarray
    best_objective: np.ndarray
    all_distances: np.ndarray  # For visualization


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
        *,
        inverse_estimators: dict[
            str, BaseEstimator | ProbabilisticEstimator | DeterministicEstimator
        ],
        forward_estimator: BaseEstimator,
        target_objective_norm: np.ndarray,
        target_objective_raw: np.ndarray,
        decisions_normalizer: BaseNormalizer,
        n_samples: int,
        distance_tolerance: float = 0.0,
    ) -> dict[str, Any]:
        """
        Runs the comparison pipeline for each inverse estimator.

        Returns:
            - results_map: Per-estimator results with selection info and distances
            - generator_runs: Pre-validation runs for visualization
        """
        results_map: dict[str, dict] = {}
        generator_runs: list[dict[str, object]] = []

        for name, estimator in inverse_estimators.items():
            # 1. Sample (Normalized space -> Physical space)

            candidates_norm = self._sample(
                estimator=estimator,
                target_objective_norm=target_objective_norm,
                n_samples=n_samples,
            )
            candidates_raw = self._denormalize(
                candidates_norm=candidates_norm,
                normalizer=decisions_normalizer,
            )

            # 2. Simulate (Predict outcomes)
            predicted_objectives = self._simulate(
                forward_estimator=forward_estimator,
                candidates_raw=candidates_raw,
            )

            # Keep pre-validation run for visualization
            generator_runs.append(
                {
                    "name": name,
                    "decisions": candidates_raw,
                    "predicted_objectives": predicted_objectives,
                }
            )

            # # 3. Validate (Optional)
            # if self._validator and self._validator.is_enabled:
            #     mask = self._validator.validate(
            #         candidates_raw=candidates_raw,
            #         predicted_objectives=predicted_objectives,
            #         target_objective_raw=target_objective_raw,
            #         distance_tolerance=distance_tolerance,
            #     )
            #     candidates_raw = candidates_raw[mask]
            #     predicted_objectives = predicted_objectives[mask]

            # 4. Select (Best candidate by distance)
            if len(candidates_raw) > 0:
                selection = self._select(
                    candidates=candidates_raw,
                    predicted_objectives=predicted_objectives,
                    target_objective_raw=target_objective_raw,
                )

                # Update generator_runs with best point info for visualization
                generator_runs[-1].update(
                    {
                        "best_index": selection.best_index,
                        "best_decision": selection.best_decision,
                        "best_objective": selection.best_objective,
                    }
                )

                results_map[name] = {
                    "decisions": candidates_raw,
                    "predicted_objectives": predicted_objectives,
                    "best_index": selection.best_index,
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

    def _normalize(
        self, candidates_norm: np.ndarray, normalizer: BaseNormalizer
    ) -> np.ndarray:
        """
        Normalizes candidate decisions from the physical space to the normalized space.
        """
        return normalizer.transform(candidates_norm)

    def _denormalize(
        self, candidates_norm: np.ndarray, normalizer: BaseNormalizer
    ) -> np.ndarray:
        """
        Denormalizes candidate decisions from the normalized space to the physical space.
        """
        return normalizer.inverse_transform(candidates_norm)

    def _simulate(
        self, forward_estimator: DeterministicEstimator, candidates_raw: np.ndarray
    ) -> np.ndarray:
        """
        Simulates the forward estimator with the candidate decisions.
        """
        return forward_estimator.predict(candidates_raw)

    def _select(
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
