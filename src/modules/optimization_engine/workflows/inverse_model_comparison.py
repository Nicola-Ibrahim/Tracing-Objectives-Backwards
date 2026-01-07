from typing import Any

import numpy as np

from ..domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
)
from ..domain.modeling.interfaces.base_normalizer import BaseNormalizer


class InverseModelEvaluator:
    """
    Domain service for evaluating a single inverse model using a forward simulator.

    This service validates an inverse model candidate against a forward model (simulator)
    to assess its quality based on multiple criteria:
    - Min Simulation Error: Lowest error achievable among samples.
    - Consistency: Median error (reliability).
    - Diversity: Spread of suggested solutions.
    - Calibration: PIT analysis against true decisions.

    Optimized with vectorized operations for batch processing.
    """

    def evaluate(
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
        Evaluates an inverse model candidate against a forward ground truth.

        Returns a dictionary containing:
        - metrics: High-level performance scores (errors, diversity, calibration).
        - ordered_candidates: Samples sorted by their simulation error (best to worst).
        - median_candidates: The component-wise median of suggested decisions.
        - distribution_stats: Statistical property of the candidate distribution.
        - calibration: Detailed PIT and CRPS data.
        """

        # 1. Pipeline: Sample -> Simulate -> Calculate Errors

        # Sample candidates from inverse model
        # Sampled candidates shape: (n_test_points, num_generated_candidates, x_dim)
        sampled_candidates = self._sample_from_inverse_model(
            inverse_estimator, test_objectives, num_samples, random_state
        )

        # Simulate forward results
        # Forward simulation results shape: (n_test_points, num_generated_candidates, y_dim)
        forward_simulation_results = self._simulate_forward_results(
            forward_estimator=forward_estimator,
            sampled_candidates=sampled_candidates,
            decision_normalizer=decision_normalizer,
            objective_normalizer=objective_normalizer,
        )

        # Calculate residuals (errors) for each sample
        # Residuals shape: (n_test_points, num_generated_candidates)
        residuals = self._calculate_residuals(
            forward_simulation_results=forward_simulation_results,
            test_objectives=test_objectives,
        )

        # 2. Extract detailed spatial information
        # Ranked candidates shape: (n_test_points, num_generated_candidates, x_dim)
        ranked_candidates = self._rank_candidates_by_residuals(
            sampled_candidates=sampled_candidates,
            residuals=residuals,
        )

        distribution_stats = self._compute_candidates_distribution_stats(
            sampled_candidates=sampled_candidates,
        )

        # 3. Aggregate performance metrics
        performance_metrics = self._compute_performance_metrics(
            residuals=residuals,
            sampled_candidates=sampled_candidates,
        )

        calibration_data = self._compute_calibration_data(
            sampled_candidates=sampled_candidates,
            test_decisions=test_decisions,
        )

        return {
            "accuracy": performance_metrics["accuracy"],
            "probabilistic": {
                "calibration_error": calibration_data["calibration_error"],
                "crps": calibration_data["crps"],
                **performance_metrics["probabilistic"],
            },
            "decisions": {
                "ordered_candidates": ranked_candidates,
                "distribution_stats": distribution_stats,
            },
            "diagnostics": {
                "raw_residuals": residuals,
                "calibration_details": calibration_data,
                **performance_metrics["distributions"],
            },
        }

    def _sample_from_inverse_model(
        self,
        inverse_estimator: BaseEstimator,
        test_objectives: np.ndarray,
        num_samples: int,
        random_state: int,
    ) -> np.ndarray:
        """
        Generates decision candidates (decisions) for the target objectives.
        Output shape: (n_test, num_samples, x_dim)
        """
        candidates = inverse_estimator.sample(
            test_objectives, n_samples=num_samples, seed=random_state
        )

        # Ensure 3D shape if only 1 sample was requested
        if candidates.ndim == 2:
            candidates = candidates[:, np.newaxis, :]

        return candidates

    def _simulate_forward_results(
        self,
        forward_estimator: BaseEstimator,
        sampled_candidates: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
    ) -> np.ndarray:
        """
        Passes candidates through the forward model to see what objectives they achieve.
        Output shape: (n_test, num_samples, y_dim)
        """
        n_test, n_samples, x_dim = sampled_candidates.shape

        # Flatten for batch processing through network
        candidates_flat = sampled_candidates.reshape(-1, x_dim)

        # Physics Space: Denormalize decisions for the simulator
        candidates_phys = decision_normalizer.inverse_transform(candidates_flat)

        # Simulation: Predict objectives in physics space
        pred_obj_phys = forward_estimator.predict(candidates_phys)
        if not isinstance(pred_obj_phys, np.ndarray):
            pred_obj_phys = np.array(pred_obj_phys)

        # Network Space: Normalize predicted objectives for error calculation
        pred_obj_norm = objective_normalizer.transform(pred_obj_phys)

        y_dim = pred_obj_norm.shape[-1]
        return pred_obj_norm.reshape(n_test, n_samples, y_dim)

    def _calculate_residuals(
        self,
        forward_simulation_results: np.ndarray,
        test_objectives: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the Euclidean distance between simulated and target objectives.
        Output shape: (n_test_points, num_generated_candidates)
        """
        # Expand targets for broadcasting: (n_test_points, 1, y_dim)
        targets_expanded = test_objectives[:, np.newaxis, :]

        # Calculate L2 norm across the objective dimension
        return np.linalg.norm(forward_simulation_results - targets_expanded, axis=2)

    def _rank_candidates_by_residuals(
        self, sampled_candidates: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """
        Sorts generated candidates for each test point from closest to farthest from target.
        """
        # Get sorting indices for each test row (n_test_points, num_generated_candidates)
        sort_indices = np.argsort(residuals, axis=1)

        # Boolean indexing for 3D is tricky, we use advanced indexing
        n_test_points, n_samples, x_dim = sampled_candidates.shape
        row_indices = np.arange(n_test_points)[:, np.newaxis]

        return sampled_candidates[row_indices, sort_indices]

    def _compute_candidates_distribution_stats(
        self, sampled_candidates: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Computes statistical properties of the suggested decision distribution.
        """
        # Median candidate: The "average" suggested solution for each test point
        median_candidates = np.median(sampled_candidates, axis=1)

        # Spread: Standard deviation per dimension
        std_candidates = np.std(sampled_candidates, axis=1)

        return {
            "median_candidates": median_candidates,
            "std_candidates": std_candidates,
        }

    def _compute_performance_metrics(
        self, residuals: np.ndarray, sampled_candidates: np.ndarray
    ) -> dict[str, Any]:
        """
        Computes high-level performance metrics for the estimator.
        """
        # Best Shot: The minimum error achieved across all samples for each test point
        # Calcualte the minimum error for each test point (n_test_points, ) row-wise
        per_point_best = np.min(residuals, axis=1)

        # Reliability: The median error across samples for each test point
        # Calculate the median error for each test point (n_test_points, ) row-wise
        per_point_reliability = np.median(residuals, axis=1)

        # Diversity: How much the suggested solutions differ from each other for each test point
        per_point_diversity = np.std(sampled_candidates, axis=1).mean(axis=1)

        # Success Rate: Percentage of points where the best shot is "accurate enough"
        # Using a threshold of 0.01 in normalized objective space (adjustable)
        threshold = 0.01
        success_rate = float(np.mean(per_point_best < threshold))

        # Quantiles of the best shot distribution across the test set
        quantiles = {
            "q25": float(np.percentile(per_point_best, 25)),
            "q50": float(np.percentile(per_point_best, 50)),  # Median
            "q75": float(np.percentile(per_point_best, 75)),
            "q95": float(np.percentile(per_point_best, 95)),
        }

        # Global aggregates (means)
        mean_best_shot = float(np.mean(per_point_best))
        mean_reliability = float(np.mean(per_point_reliability))
        mean_diversity = float(np.mean(per_point_diversity))

        # Sharpness: The width of the predictive distribution (90% interval)
        q95_dec = np.percentile(sampled_candidates, 95, axis=1)
        q05_dec = np.percentile(sampled_candidates, 5, axis=1)
        mean_sharpness = float(np.mean(q95_dec - q05_dec))

        return {
            "accuracy": {
                "mean_best_shot_error": mean_best_shot,
                "median_best_shot_error": quantiles["q50"],
                "reliability_error": mean_reliability,
                "success_rate_0.01": success_rate,
                "quantiles": quantiles,
            },
            "probabilistic": {
                "diversity_score": mean_diversity,
                "sharpness": mean_sharpness,
            },
            "distributions": {
                "best_shots": per_point_best,
                "reliabilities": per_point_reliability,
                "diversities": per_point_diversity,
            },
        }

    def _compute_calibration_data(
        self, sampled_candidates: np.ndarray, test_decisions: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyzes how well the model's confidence matches its actual accuracy.
        Uses PIT (Probability Integral Transform) and CRPS (Continuous Ranked Probability Score).
        """
        n_test, n_samples, x_dim = sampled_candidates.shape
        pit_values = []
        crps_per_dim = []

        for i in range(n_test):
            for d in range(x_dim):
                true_val = test_decisions[i, d]
                model_samples = sampled_candidates[i, :, d]

                # PIT: Fraction of samples lower than the true value.
                # If calibrated, PIT should follow a Uniform distribution.
                pit = np.mean(model_samples <= true_val)
                pit_values.append(pit)

                # CRPS: Measures the distance between the predictive CDF and the true value (step function).
                # Formula: E|X - y| - 0.5 * E|X - X'|
                mae_to_true = np.mean(np.abs(model_samples - true_val))
                diff_matrix = np.abs(model_samples[:, None] - model_samples)
                expected_dist = np.mean(diff_matrix)

                crps_val = mae_to_true - 0.5 * expected_dist
                crps_per_dim.append(crps_val)

        pit_values = np.sort(pit_values)
        cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)

        # Calibration Error: Mean absolute deviation from the ideal uniform CDF
        calibration_error = float(np.mean(np.abs(pit_values - cdf_y)))
        mean_crps = float(np.mean(crps_per_dim))

        return {
            "pit_values": pit_values,
            "cdf_y": cdf_y,
            "calibration_error": calibration_error,
            "crps": mean_crps,
        }


class InverseModelComparator:
    """
    Workflow for comparing multiple inverse models.
    """

    def __init__(self, evaluator: InverseModelEvaluator | None = None) -> None:
        self._evaluator = evaluator or InverseModelEvaluator()

    def compare(
        self,
        forward_estimator: BaseEstimator,
        inverse_estimators: dict[str, BaseEstimator],
        test_objectives: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
        test_decisions: np.ndarray,
        num_samples: int = 250,
        random_state: int = 42,
    ) -> dict[str, dict[str, Any]]:
        """
        Compares multiple inverse estimators and returns a structured dictionary.
        """
        comparison_results = {}

        for name, estimator in inverse_estimators.items():
            results = self._evaluator.evaluate(
                inverse_estimator=estimator,
                forward_estimator=forward_estimator,
                test_objectives=test_objectives,
                decision_normalizer=decision_normalizer,
                objective_normalizer=objective_normalizer,
                test_decisions=test_decisions,
                num_samples=num_samples,
                random_state=random_state,
            )
            comparison_results[name] = results

        return comparison_results
