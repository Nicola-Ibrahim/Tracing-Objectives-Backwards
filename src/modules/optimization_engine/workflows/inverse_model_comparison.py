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
        """
        Main entry point for inverse model validation.
        Samples candidates, runs them through the simulator, and computes metrics.
        """
        # 1. Sample and simulate candidates using the forward model
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

        # 2. Compute performance metrics (Best Shot, Reliability, Diversity)
        metrics = self._compute_performance_metrics(
            raw_errors=raw_errors, candidates_3d=candidates_3d
        )

        # 3. Compute calibration data (PIT)
        calibration = self._compute_calibration_data(
            candidates_3d=candidates_3d, test_decisions=test_decisions
        )

        return {
            "metrics": {
                "best_shot_error": metrics["best_shot_error"],
                "calibration_error": calibration["calibration_error"],
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
        """
        Aggregates error matrix into scalar performance metrics.
        """
        # 1. Best Shot (Min error per target) -> Mean across targets
        best_shots = np.min(raw_errors, axis=1)
        mean_best_shot = float(np.mean(best_shots))

        # 2. Reliability (Median error per target) -> Mean across targets
        reliabilities = np.median(raw_errors, axis=1)
        mean_reliability = float(np.mean(reliabilities))

        # 3. Diversity (Std Dev of candidates per target) -> Mean across targets
        # std along sample axis (axis=1), then mean over dimensions (axis=2)
        diversities = np.std(candidates_3d, axis=1).mean(axis=1)
        mean_diversity = float(np.mean(diversities))

        return {
            "best_shot_error": mean_best_shot,
            "average_median_error": mean_reliability,
            "diversity_score": mean_diversity,
        }

    def _compute_calibration_data(
        self, candidates_3d: np.ndarray, test_decisions: np.ndarray
    ) -> dict[str, Any]:
        """
        Computes PIT (Probability Integral Transform) for calibration plotting
        and calculates the Calibration Error (Expected Calibration Error).
        """
        n_test, n_samples, x_dim = candidates_3d.shape
        pit_values = []

        # Calculate PIT for each dimension of each test sample
        for i in range(n_test):
            for d in range(x_dim):
                true_val = test_decisions[i, d]
                model_samples = candidates_3d[i, :, d]
                pit = np.mean(model_samples <= true_val)
                pit_values.append(pit)

        pit_values = np.sort(pit_values)

        # Calculate empirical CDF values for PIT values
        # This is simply the rank divided by total number of values
        cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)

        # Calculate Calibration Error (Expected Calibration Error)
        # Average Absolute Deviation from the Diagonal
        calibration_error = float(np.mean(np.abs(pit_values - cdf_y)))

        return {
            "pit_values": pit_values,
            "cdf_y": cdf_y,
            "calibration_error": calibration_error,
        }
