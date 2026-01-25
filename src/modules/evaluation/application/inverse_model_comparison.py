from typing import Any

import numpy as np

from modules.modeling.domain.interfaces.base_estimator import (
    BaseEstimator,
)
from modules.modeling.domain.interfaces.base_normalizer import BaseNormalizer


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
        test_decisions: np.ndarray,
        decision_normalizer: BaseNormalizer,
        objective_normalizer: BaseNormalizer,
        num_samples: int,
    ) -> dict[str, Any]:
        """
        Evaluates an inverse model candidate against a forward ground truth.

        Returns a dictionary containing:
        - metrics:
            - accuracy: mean_lowest_resi_residual, median_best_shot_residual, reliability_residual, etc.
            - uncertainty: diversity_score, interval_width.
            - calibration: calibration_residual, crps.
        - detailed_results:
            - residuals: lowest_residual, reliability, diversity, all_samples.
            - candidates: ordered, median, std.
            - calibration_curves: pit_values, cdf_y.
        """

        # 1. Pipeline: Sample -> Simulate -> Calculate Errors

        # Sample candidates from inverse model
        # Sampled candidates shape: (n_test_samples, num_generated_candidates, x_dim)
        sampled_candidates = self._sample_from_inverse_model(
            inverse_estimator, test_objectives, num_samples
        )

        # Simulate forward results
        # Forward simulation results shape: (n_test_samples, num_generated_candidates, y_dim)
        forward_simulation_results = self._simulate_forward_results(
            forward_estimator=forward_estimator,
            sampled_candidates=sampled_candidates,
            decision_normalizer=decision_normalizer,
            objective_normalizer=objective_normalizer,
        )

        # Calculate residuals (errors) for each sample
        # Residuals shape: (n_test_samples, num_generated_candidates)
        residuals = self._calculate_residuals(
            forward_simulation_results=forward_simulation_results,
            test_objectives=test_objectives,
        )

        # 2. Extract detailed spatial information
        # Ranked candidates shape: (n_test_samples, num_generated_candidates, x_dim)
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
            "performance": performance_metrics,
            "candidates": {
                "ordered": ranked_candidates,
                "median": distribution_stats["median_candidates"],
                "std": distribution_stats["std_candidates"],
            },
            "calibration": calibration_data,
        }

    def _sample_from_inverse_model(
        self,
        inverse_estimator: BaseEstimator,
        test_objectives: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """
        Generates decision candidates (decisions) for the target objectives.
        Output shape: (n_test, num_samples, x_dim)
        """
        candidates = inverse_estimator.sample(test_objectives, n_samples=num_samples)

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
        Output shape: (n_test_samples, num_generated_candidates)
        """
        # Expand targets for broadcasting: (n_test_samples, 1, y_dim)
        targets_expanded = test_objectives[:, np.newaxis, :]

        # Calculate L2 norm across the objective dimension
        return np.linalg.norm(forward_simulation_results - targets_expanded, axis=2)

    def _rank_candidates_by_residuals(
        self, sampled_candidates: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """
        Sorts generated candidates for each test point from closest to farthest from target.
        """
        # Get sorting indices for each test row (n_test_samples, num_generated_candidates)
        sort_indices = np.argsort(residuals, axis=1)

        # Boolean indexing for 3D is tricky, we use advanced indexing
        n_test_samples, n_samples, x_dim = sampled_candidates.shape
        row_indices = np.arange(n_test_samples)[:, np.newaxis]

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
        # The shape is (n_test_samples, ) row-wise
        lowest_residual = np.min(residuals, axis=1)

        # Reliability: The median error across samples for each test point
        # The shape is (n_test_samples, ) row-wise
        reliability = np.median(residuals, axis=1)

        # Diversity: How much the suggested solutions differ from each other for each test point
        # The shape is (n_test_samples, ) row-wise
        diversity = np.std(sampled_candidates, axis=1).mean(axis=1)

        # Success Rate: Percentage of points where the best shot is "accurate enough"
        # Using a threshold of 0.01 in normalized objective space (adjustable)
        threshold = 0.01
        success_rate = float(np.mean(lowest_residual < threshold))

        # Quantiles of the best shot distribution across the test set
        quantiles = {
            "q25": float(np.percentile(lowest_residual, 25)),
            "q50": float(np.percentile(lowest_residual, 50)),  # Median
            "q75": float(np.percentile(lowest_residual, 75)),
            "q95": float(np.percentile(lowest_residual, 95)),
        }

        # Global aggregates (means)
        mean_lowest_residual = float(np.mean(lowest_residual))
        median_lowest_residual = float(np.median(lowest_residual))
        mean_reliability = float(np.mean(reliability))
        mean_diversity = float(np.mean(diversity))

        # Interval Width (90%): The width of the predictive distribution
        q95_dec = np.percentile(sampled_candidates, 95, axis=1)
        q05_dec = np.percentile(sampled_candidates, 5, axis=1)
        interval_width = q95_dec - q05_dec
        mean_interval_width = float(np.mean(interval_width))

        return {
            "mean_lowest_residual": mean_lowest_residual,
            "median_lowest_residual": median_lowest_residual,
            "mean_reliability": mean_reliability,
            "success_rate": success_rate,
            "quantiles": quantiles,
            "mean_diversity": mean_diversity,
            "mean_interval_width": mean_interval_width,
            "distributions": {
                "lowest_residual": lowest_residual,
                "reliability": reliability,
                "diversity": diversity,
                "interval_width": interval_width,
            },
        }

    def _compute_calibration_data(
        self, sampled_candidates: np.ndarray, test_decisions: np.ndarray
    ) -> dict[str, Any]:
        """
        Evaluates the model's performance using two distinct lenses:
        1. CRPS (How useful/accurate is the guess?)
        2. Calibration (How honest is the model about its uncertainty?)

        --- CRPS (Continuous Ranked Probability Score) ---
        Think of this as the "Total Performance" grade.
        It rewards models that are both 'Accurate' (near the truth) and 'Sharp'
        (not just guessing wildly).
        - A lower CRPS is better.
        - Formula used: E|X - y| (Distance to truth) - 0.5 * E|X - X'| (Spread of guesses).

        --- Calibration & PIT (Probability Integral Transform) ---
        This measures "Honesty." It doesn't care if the model missed the target;
        it cares if the model *knew* it might miss.
        - PIT Values: For every test point, we see where the truth fell relative
        to our samples.
        - PIT Curve: If calibrated, this should be a diagonal line (Uniform).
        - U-Shape curve: Model is 'Overconfident' (Truth falls outside its predicted range often).
        - Hump-Shape curve: Model is 'Underconfident' (Truth is always in the middle, but model predicted a huge range).

        Returns:
            A dictionary containing the calibration error (gap from honesty),
            mean CRPS (overall quality), and the curve data for plotting.
        """
        n_test, n_samples, x_dim = sampled_candidates.shape
        pit_values = []
        crps_per_dim = []

        for i in range(n_test):
            for d in range(x_dim):
                true_val = test_decisions[i, d]
                model_samples = sampled_candidates[i, :, d]

                # 1. Calculate PIT (For the "Honesty" check)
                # Find what 'percentile' the true value landed in.
                pit = np.mean(model_samples <= true_val)
                pit_values.append(pit)

                # 2. Calculate CRPS (For the "Total Grade")
                # Part A: Average distance of samples to the truth (Accuracy)
                mae_to_true = np.mean(np.abs(model_samples - true_val))
                # Part B: Average distance between samples (Sharpness/Certainty)
                diff_matrix = np.abs(model_samples[:, None] - model_samples)
                expected_dist = np.mean(diff_matrix)

                # Total score = Accuracy - 0.5 * Sharpness
                crps_val = mae_to_true - 0.5 * expected_dist
                crps_per_dim.append(crps_val)

        # Sort PIT values to create the calibration curve
        pit_values = np.sort(pit_values)
        cdf_y = np.arange(1, len(pit_values) + 1) / len(pit_values)

        # Calibration Error: How far is our PIT curve from the ideal diagonal?
        calibration_error = float(np.mean(np.abs(pit_values - cdf_y)))
        mean_crps = float(np.mean(crps_per_dim))

        return {
            "calibration_error": calibration_error,
            "mean_crps": mean_crps,
            "curve": {
                "pit_values": pit_values,
                "cdf_y": cdf_y,
            },
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
        num_samples: int,
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
            )
            comparison_results[name] = results

        return comparison_results
