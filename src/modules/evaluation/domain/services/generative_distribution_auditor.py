from dataclasses import dataclass

import numpy as np

from ..value_objects.calibration_curve import CalibrationCurve
from ..value_objects.reliability_summary import ReliabilitySummary


@dataclass(frozen=True)
class GenerativeAudit:
    """The immutable value object representing the outcome of a generative distribution audit."""

    pit_values: np.ndarray  # (N_test * x_dim,)
    calibration_error: float  # MACE
    crps: float
    diversity: np.ndarray  # (N_test,)
    interval_width: np.ndarray  # (N_test, x_dim)
    summary: ReliabilitySummary
    calibration_curve: CalibrationCurve


class GenerativeDistributionAuditor:
    """
    Domain Service: Audits the fidelity and reliability of generated distributions.
    Calculates calibration (PIT, MACE), sharpness (CRPS, Interval Width), and diversity.
    """

    @staticmethod
    def audit(samples: np.ndarray, truth: np.ndarray) -> GenerativeAudit:
        """
        Audits the generated solution cloud against the ground truth.

        Args:
            samples: (N_test, K_samples, x_dim)
            truth: (N_test, x_dim)
        """
        n_test, k_samples, x_dim = samples.shape

        # 1. CALIBRATION (PIT and MACE)
        pit_values = GenerativeDistributionAuditor._compute_pit_values(samples, truth)
        mace = GenerativeDistributionAuditor._compute_calibration_error(pit_values)

        # 2. PERFORMANCE (CRPS)
        crps = GenerativeDistributionAuditor._compute_crps(samples, truth)

        # 3. DISTRIBUTION STATS (Diversity and Interval Width)
        diversity = np.std(samples, axis=1).mean(axis=1)

        quantile_width = 0.90
        q_high = (1 + quantile_width) / 2
        q_low = (1 - quantile_width) / 2
        q95 = np.percentile(samples, q_high * 100, axis=1)
        q05 = np.percentile(samples, q_low * 100, axis=1)
        intervals = q95 - q05

        # 4. SUMMARY and CURVES
        summary = ReliabilitySummary(
            mean_crps=crps,
            mean_diversity=float(np.mean(diversity)),
            mean_interval_width=float(np.mean(intervals)),
        )

        calibration_curve = CalibrationCurve(
            pit_values=np.sort(pit_values),
            cdf_y=np.arange(1, len(pit_values) + 1) / len(pit_values),
        )

        return GenerativeAudit(
            pit_values=pit_values,
            calibration_error=mace,
            crps=crps,
            diversity=diversity,
            interval_width=intervals,
            summary=summary,
            calibration_curve=calibration_curve,
        )

    @staticmethod
    def _compute_pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
        n_test, k_samples, x_dim = samples.shape
        pit_results = []
        for i in range(n_test):
            for d in range(x_dim):
                true_val = truth[i, d]
                model_samples = samples[i, :, d]
                pit = np.mean(model_samples <= true_val)
                pit_results.append(pit)
        return np.array(pit_results)

    @staticmethod
    def _compute_calibration_error(pit_values: np.ndarray) -> float:
        sorted_pit = np.sort(pit_values)
        cdf_y = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)
        return float(np.mean(np.abs(sorted_pit - cdf_y)))

    @staticmethod
    def _compute_crps(samples: np.ndarray, truth: np.ndarray) -> float:
        n_test, k_samples, x_dim = samples.shape
        crps_per_dim = []
        for i in range(n_test):
            for d in range(x_dim):
                true_val = truth[i, d]
                model_samples = samples[i, :, d]
                mae_to_true = np.mean(np.abs(model_samples - true_val))
                diff_matrix = np.abs(model_samples[:, np.newaxis] - model_samples)
                expected_dist = np.mean(diff_matrix)
                crps_val = mae_to_true - 0.5 * expected_dist
                crps_per_dim.append(crps_val)
        return float(np.mean(crps_per_dim))
