from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SpatialAudit:
    """The immutable value object representing the outcome of a spatial audit."""

    residuals: np.ndarray  # r_j^(k): (N, K, M)
    discrepancy_scores: np.ndarray  # s^(k): (N, K)
    rank_indices: np.ndarray  # Sorted indices (N, K)
    bias: np.ndarray  # b(y*): (N,)
    dispersion: np.ndarray  # v(y*): (N,)

    @property
    def best_index(self) -> np.ndarray:
        """Helper to identify the 'winner' for each test point."""
        return self.rank_indices[:, 0]


class SpatialCandidateAuditor:
    """
    Domain Service: Implements the Standardized Discrepancy Contract.
    Calculates residuals, rankings, and cloud-plausibility diagnostics.
    """

    @staticmethod
    def audit(
        candidates: np.ndarray,
        reference: np.ndarray,
        tau: np.ndarray,
        precision_matrix: Optional[np.ndarray] = None,
    ) -> SpatialAudit:
        """
        Processes a cloud of candidates relative to a reference (target or truth).

        Args:
            candidates: (N, K, M) array of predicted outcomes.
            reference: (N, M) array of targets.
            tau: (M,) scale vector (SD, MAD, or IQR).
            precision_matrix: (M, M) Inverse Covariance (Sigma^-1) for Mahalanobis.
        """
        n_points, k_samples, m_dims = candidates.shape

        # 1. STANDARDIZED RESIDUALS (Subtract and Divide)
        # r = (y_hat - y_star) / tau
        residuals = (candidates - reference[:, np.newaxis, :]) / tau

        # 2. DISCREPANCY SCORING (The Contract)
        if precision_matrix is not None:
            # Mahalanobis: sqrt( r.T * Sigma^-1 * r )
            # We use einsum for efficient batch matrix multiplication
            # 'nkm,mj,nkj->nk' handles (N, K, M) dots (M, M) dots (N, K, M)
            sq_dist = np.einsum(
                "nkm,mj,nkj->nk", residuals, precision_matrix, residuals
            )
            discrepancy_scores = np.sqrt(np.maximum(sq_dist, 0))
        else:
            # Euclidean Default: L2 Norm of residuals
            discrepancy_scores = np.linalg.norm(residuals, axis=2)

        # 3. RANKING
        rank_indices = np.argsort(discrepancy_scores, axis=1)

        # 4. CLOUD DIAGNOSTICS (Bias and Dispersion)
        # Systematic Bias: Magnitude of mean residual vector / sqrt(m)
        mean_residual = np.mean(residuals, axis=1)
        bias = np.linalg.norm(mean_residual, axis=1) / np.sqrt(m_dims)

        # Cloud Dispersion: Median distance from the mean residual / sqrt(m)
        centered_residuals = residuals - mean_residual[:, np.newaxis, :]
        dist_to_mean = np.linalg.norm(centered_residuals, axis=2)
        dispersion = np.median(dist_to_mean, axis=1) / np.sqrt(m_dims)

        return SpatialAudit(
            residuals=residuals,
            discrepancy_scores=discrepancy_scores,
            rank_indices=rank_indices,
            bias=bias,
            dispersion=dispersion,
        )
