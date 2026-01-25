from typing import Any

import numpy as np

from modules.modeling.domain.interfaces.base_estimator import BaseEstimator
from modules.modeling.domain.interfaces.base_normalizer import BaseNormalizer

from .components import (
    CandidateSelector,
    CandidateValidator,
    DecisionSampler,
    ForwardSimulator,
)


class InverseGeneratorComparator:
    """
    Compares multiple inverse estimators by observing how each generates
    candidate decisions for a target objective.

    Pipeline per estimator:
    1. Sample: Generate candidate decisions from inverse estimator
    2. Simulate: Predict objectives via forward model
    3. Validate: (Optional) Filter candidates using OOD/Conformal validators
    4. Select: Pick best candidate by distance to target
    """

    def __init__(
        self,
        sampler: DecisionSampler | None = None,
        simulator: ForwardSimulator | None = None,
        validator: CandidateValidator | None = None,
        selector: CandidateSelector | None = None,
    ) -> None:
        self._sampler = sampler or DecisionSampler()
        self._simulator = simulator or ForwardSimulator()
        self._validator = validator  # Can be None if no validation is desired
        self._selector = selector or CandidateSelector()

    def compare(
        self,
        *,
        inverse_estimators: dict[str, BaseEstimator],
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
            candidates_norm = self._sampler.sample(
                estimator=estimator,
                target_objective_norm=target_objective_norm,
                n_samples=n_samples,
            )
            candidates_raw = self._sampler.denormalize(
                candidates_norm=candidates_norm,
                normalizer=decisions_normalizer,
            )

            # 2. Simulate (Predict outcomes)
            predicted_objectives = self._simulator.predict(
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

            # 3. Validate (Optional)
            if self._validator and self._validator.is_enabled:
                mask = self._validator.validate(
                    candidates_raw=candidates_raw,
                    predicted_objectives=predicted_objectives,
                    target_objective_raw=target_objective_raw,
                    distance_tolerance=distance_tolerance,
                )
                candidates_raw = candidates_raw[mask]
                predicted_objectives = predicted_objectives[mask]

            # 4. Select (Best candidate by distance)
            if len(candidates_raw) > 0:
                selection = self._selector.select(
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
