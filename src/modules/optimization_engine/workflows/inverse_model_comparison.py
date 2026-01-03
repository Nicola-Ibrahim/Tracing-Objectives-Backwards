from typing import Any

import numpy as np

from ..domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
)
from ..domain.modeling.interfaces.base_normalizer import BaseNormalizer


class InverseModelComparator:
    """
    Domain service for comparing inverse models using a forward simulator.

    This service validates inverse model candidates against a forward model (simulator)
    to assess their quality based on multiple criteria:
    - Min Simulation Error: Lowest error achievable among samples.
    - Consistency: Median error (reliability).
    - Diversity: Spread of suggested solutions.
    - Calibration: PIT analysis against true decisions.

    Optimized with vectorized operations for batch processing.
    """

    def validate(
        self,
        inverse_estimator: BaseEstimator,
        forward_estimator: BaseEstimator,
        test_objectives: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
        test_decisions: np.ndarray,
        num_samples: int = 250,
        random_state: int = 42,
    ) -> dict[str, Any]:
        simulation_results = self._generate_and_simulate_candidates(
            inverse_estimator=inverse_estimator,
            forward_estimator=forward_estimator,
            test_objectives=test_objectives,
            decision_normalizer=decision_normalizer,
            objective_normalizer=objective_normalizer,
            num_samples=num_samples,
            random_state=random_state,
        )

        candidates_3d = simulation_results["candidates_3d"]
        raw_errors = simulation_results["raw_errors"]

        metrics = self._compute_performance_metrics(
            raw_errors=raw_errors, candidates_3d=candidates_3d
        )

        calibration = self._compute_calibration_data(
            candidates_3d=candidates_3d, test_decisions=test_decisions
        )

        return {
            "metrics": {
                "best_shot_error": metrics["best_shot_error"],
                "calibration_error": calibration["calibration_error"],
                "sharpness": metrics["sharpness"],
                "crps": calibration["crps"],
                "diversity_score": metrics["diversity_score"],
            },
            "raw_errors": raw_errors,
            "calibration": calibration,
        }

    def _generate_and_simulate_candidates(
        self,
        inverse_estimator: BaseEstimator,
        forward_estimator: BaseEstimator,
        test_objectives: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
        num_samples: int,
        random_state: int,
    ) -> dict[str, np.ndarray]:
        """
        Generates candidates from inverse model and runs them through the forward model.
        Returns the raw error matrix (n_test, num_samples).
        """
        n_test, y_dim = test_objectives.shape

        # STEP 1: Vectorized Sampling
        # We ask the estimator to sample for ALL test targets at once.
        # Expected Output Shape: (n_test, num_samples, x_dim)
        candidates_3d = inverse_estimator.sample(
            test_objectives, n_samples=num_samples, seed=random_state
        )
        if candidates_3d.ndim == 2:
            candidates_3d = candidates_3d[:, np.newaxis, :]

        n_test_act, n_samp_act, x_dim = candidates_3d.shape

        # STEP 2: Batch Simulation
        # Flatten for processing through normalizers and forward model
        candidates_flat = candidates_3d.reshape(-1, x_dim)

        # A. Denormalize decisions (Network Space -> Physics Space)
        candidates_orig = decision_normalizer.inverse_transform(candidates_flat)

        # B. Run Simulator / Forward Model
        pred_obj_flat = forward_estimator.predict(candidates_orig)

        if not isinstance(pred_obj_flat, np.ndarray):
            pred_obj_flat = np.array(pred_obj_flat)

        # C. Normalize predicted objectives (Physics Space -> Normalized Space)
        pred_obj_norm = objective_normalizer.transform(pred_obj_flat)

        # STEP 3: Reshape back to (n_test, num_samples, y_dim)
        pred_obj_norm_3d = pred_obj_norm.reshape(n_test_act, n_samp_act, y_dim)

        # STEP 4: Calculate Euclidean Distance (Error)
        # Prepare targets for broadcasting: (n_test, 1, y_dim)
        targets_expanded = test_objectives[:, np.newaxis, :]
        raw_errors = np.linalg.norm(pred_obj_norm_3d - targets_expanded, axis=2)

        return {
            "candidates_3d": candidates_3d,
            "raw_errors": raw_errors,
        }

    def _compute_performance_metrics(
        self, raw_errors: np.ndarray, candidates_3d: np.ndarray
    ) -> dict[str, float]:
        best_shots = np.min(raw_errors, axis=1)
        mean_best_shot = float(np.mean(best_shots))

        reliabilities = np.median(raw_errors, axis=1)
        mean_reliability = float(np.mean(reliabilities))

        diversities = np.std(candidates_3d, axis=1).mean(axis=1)
        mean_diversity = float(np.mean(diversities))

        # 4. Sharpness: Width of the 90% Prediction Interval
        # Average (95th percentile - 5th percentile) across samples and dimensions
        q95 = np.percentile(candidates_3d, 95, axis=1)
        q05 = np.percentile(candidates_3d, 5, axis=1)
        sharpness = float(np.mean(q95 - q05))

        return {
            "best_shot_error": mean_best_shot,
            "average_median_error": mean_reliability,
            "diversity_score": mean_diversity,
            "sharpness": sharpness,
        }

    def _compute_calibration_data(
        self, candidates_3d: np.ndarray, test_decisions: np.ndarray
    ) -> dict[str, Any]:
        n_test, n_samples, x_dim = candidates_3d.shape
        pit_values = []
        crps_per_dim = []

        for i in range(n_test):
            for d in range(x_dim):
                true_val = test_decisions[i, d]
                model_samples = candidates_3d[i, :, d]

                # PIT Calculation
                pit = np.mean(model_samples <= true_val)
                pit_values.append(pit)

                # 5. CRPS (Sample-based approximation)
                # Formula: E|X - y| - 0.5 * E|X - X'|
                mae_to_true = np.mean(np.abs(model_samples - true_val))

                # Pairwise absolute difference between all samples
                # Note: For very large n_samples, consider a more optimized version
                diff_matrix = np.abs(model_samples[:, None] - model_samples)
                expected_dist = np.mean(diff_matrix)

                crps_val = mae_to_true - 0.5 * expected_dist
                crps_per_dim.append(crps_val)

        pit_values = np.sort(pit_values)
        cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)
        calibration_error = float(np.mean(np.abs(pit_values - cdf_y)))
        mean_crps = float(np.mean(crps_per_dim))

        return {
            "pit_values": pit_values,
            "cdf_y": cdf_y,
            "calibration_error": calibration_error,
            "crps": mean_crps,
        }
