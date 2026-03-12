import numpy as np
from ..value_objects.decision_assessment import (
    DecisionSpaceDistributionAssessment,
    DecisionSpaceIntervalAssessment,
)
from ..value_objects.calibration_curve import CalibrationCurve
from ..value_objects.pit_profile import PITProfile
from ..value_objects.ecdf_profile import ECDFProfile
from ..enums.engine_capability import EngineCapability


class DecisionSpaceAuditor:
    """
    Domain Service: Assessment in decision space (probabilistic/calibration).
    Handles different engine capabilities: FULL_DISTRIBUTION vs PREDICTION_INTERVAL.
    """

    @staticmethod
    def audit(
        capability: EngineCapability,
        samples: np.ndarray,
        truth: np.ndarray,
    ) -> DecisionSpaceDistributionAssessment | DecisionSpaceIntervalAssessment:
        """
        Calculates probabilistic metrics in decision space.

        Args:
            capability: Enum indicating engine type.
            samples: (N, K, D) generated samples/predictions.
            truth: (N, D) ground truth decisions.
        """
        if capability == EngineCapability.FULL_DISTRIBUTION:
            return DecisionSpaceAuditor._audit_distributional(samples, truth)
        elif capability == EngineCapability.PREDICTION_INTERVAL:
            return DecisionSpaceAuditor._audit_interval(samples, truth)
        else:
            raise ValueError(f"Unsupported engine capability: {capability}")

    @staticmethod
    def _audit_distributional(
        samples: np.ndarray, truth: np.ndarray
    ) -> DecisionSpaceDistributionAssessment:
        n_test, k_samples, d_dims = samples.shape

        # 1. PIT Values (marginal PIT across all dimensions)
        pit_values = DecisionSpaceAuditor._compute_pit_values(samples, truth)
        
        # 2. MACE (Mean Absolute Calibration Error)
        mace = DecisionSpaceAuditor._compute_mace(pit_values)

        # 3. CRPS (Continuous Ranked Probability Score)
        crps = DecisionSpaceAuditor._compute_crps(samples, truth)

        # 4. Diversity and Interval Width
        diversity = np.mean(np.std(samples, axis=1)) # Average of std across K per sample
        
        # 90% interval width
        q95 = np.percentile(samples, 95, axis=1)
        q05 = np.percentile(samples, 5, axis=1)
        mean_interval_width = float(np.mean(q95 - q05))

        # 5. Profiles
        counts, bin_edges = np.histogram(pit_values, bins=10, range=(0, 1))
        pit_profile = PITProfile(bin_edges=bin_edges.tolist(), counts=counts.tolist())

        # Calibration Curve (Diagonal = Perfect)
        nominal_coverage = np.linspace(0, 1, 11)
        empirical_coverage = [np.mean(pit_values <= q) for q in nominal_coverage]
        calibration_curve = CalibrationCurve(
            nominal_coverage=nominal_coverage.tolist(),
            empirical_coverage=empirical_coverage
        )

        return DecisionSpaceDistributionAssessment(
            pit_profile=pit_profile,
            calibration_curve=calibration_curve,
            mace=mace,
            mean_crps=crps,
            mean_interval_width=mean_interval_width,
            mean_diversity=float(diversity),
        )

    @staticmethod
    def _audit_interval(
        samples: np.ndarray, truth: np.ndarray
    ) -> DecisionSpaceIntervalAssessment:
        # For interval engines, we assume samples contain at least min and max
        n_test, k_samples, d_dims = samples.shape
        
        lower = np.min(samples, axis=1)  # (N, D)
        upper = np.max(samples, axis=1)  # (N, D)
        
        # 1. Empirical Coverage at specific levels
        # If we only have ONE interval (e.g. 95%), we can only check that level.
        # But ECDFProfile suggests we want a curve.
        # If we have K samples, we can check coverage at multiple quantiles.
        nominal_levels = np.linspace(0.1, 0.9, 9)
        emp_coverage = []
        for level in nominal_levels:
            q_low = (1 - level) / 2
            q_high = (1 + level) / 2
            low_b = np.percentile(samples, q_low * 100, axis=1)
            high_b = np.percentile(samples, q_high * 100, axis=1)
            inside = (truth >= low_b) & (truth <= high_b)
            emp_coverage.append(float(np.mean(inside)))
            
        ecdf_profile = ECDFProfile(
            x_values=nominal_levels.tolist(),
            cumulative_probabilities=emp_coverage
        )
        
        # 2. Mean Coverage Error
        mean_coverage_error = float(np.mean(np.abs(nominal_levels - emp_coverage)))
        
        # 3. Interval Width (using the outermost range provided)
        mean_interval_width = float(np.mean(upper - lower))
        
        # 4. Winkler Score (at 90% nominal coverage)
        alpha = 0.1
        outside_low = np.maximum(lower - truth, 0)
        outside_high = np.maximum(truth - upper, 0)
        winkler_scores = (upper - lower) + (2 / alpha) * (outside_low + outside_high)
        mean_winkler_score = float(np.mean(winkler_scores))

        return DecisionSpaceIntervalAssessment(
            ecdf_profile=ecdf_profile,
            mean_coverage_error=mean_coverage_error,
            mean_interval_width=mean_interval_width,
            mean_winkler_score=mean_winkler_score,
        )

    @staticmethod
    def _compute_pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
        n_test, k_samples, d_dims = samples.shape
        pit_results = []
        for i in range(n_test):
            for d in range(d_dims):
                true_val = truth[i, d]
                model_samples = samples[i, :, d]
                # Small noise to handle ties if needed, but mean is standard
                pit = np.mean(model_samples <= true_val)
                pit_results.append(pit)
        return np.array(pit_results)

    @staticmethod
    def _compute_mace(pit_values: np.ndarray) -> float:
        sorted_pit = np.sort(pit_values)
        cdf_y = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)
        return float(np.mean(np.abs(sorted_pit - cdf_y)))

    @staticmethod
    def _compute_crps(samples: np.ndarray, truth: np.ndarray) -> float:
        # Standard U-statistic estimator for CRPS
        n_test, k_samples, d_dims = samples.shape
        crps_per_dim = []
        for i in range(n_test):
            for d in range(d_dims):
                true_val = truth[i, d]
                model_samples = samples[i, :, d]
                mae_to_true = np.mean(np.abs(model_samples - true_val))
                diff_matrix = np.abs(model_samples[:, np.newaxis] - model_samples)
                expected_dist = np.mean(diff_matrix)
                crps_val = mae_to_true - 0.5 * expected_dist
                crps_per_dim.append(crps_val)
        return float(np.mean(crps_per_dim))
