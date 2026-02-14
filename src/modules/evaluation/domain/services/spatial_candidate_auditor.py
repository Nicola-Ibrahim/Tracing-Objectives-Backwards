from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..value_objects.accuracy_summary import AccuracySummary

DistanceType = Literal["euclidean", "mahalanobis"]


@dataclass(frozen=True)
class SpatialAudit:
    """Immutable result of the objective-space audit (normalized objective space)."""

    # N: number of targets
    # K: number of candidates
    # M: number of objectives
    residuals: np.ndarray  # r^(k)(y*): (N, K, M)  = y_hat^(k) - y*
    discrepancy_scores: np.ndarray  # s^(k)(y*): (N, K)
    best_shot_scores: np.ndarray  # min_k(s): (N,)
    rank_indices: np.ndarray  # argsort(s): (N, K), ascending (best first)
    bias: np.ndarray  # b(y*): (N,)
    dispersion: np.ndarray  # v(y*): (N,)
    summary: AccuracySummary

    @property
    def best_index(self) -> np.ndarray:
        """Index of the best-ranked candidate per target (shape: (N,))."""
        return self.rank_indices[:, 0]


class SpatialCandidateAuditor:
    """
    Domain Service: Discrepancy contract in *normalized* objective space.

    Assumption:
      - candidates and reference live in the same normalized objective space
        (e.g., HypercubeNormalizer [0,1] using training-fitted min/max).
      - Because objectives are already on a common scale, residuals are defined
        as simple differences (no additional tau scaling).

    Policy (simple, defensible):
      - Euclidean discrepancy is the default.
      - Mahalanobis discrepancy is optional and estimated from the training
        objective archive in the same normalized space.
    """

    @staticmethod
    def audit(
        training_objectives: np.ndarray,
        candidates: np.ndarray,
        reference: np.ndarray,
        distance: DistanceType = "euclidean",
        ridge: float = 1e-6,
    ) -> SpatialAudit:
        """

        Args:
            training_objectives: (N_tr, M) training outcomes in the same normalized space.
            candidates: (N, K, M) checked candidate outcomes (y_hat) in normalized space.
            reference: (N, M) target outcomes (y*) in normalized space.
            distance: "euclidean" | "mahalanobis".
            ridge: nonnegative diagonal regularizer added to covariance before inversion.
        """
        # -----------------------------
        # 0) Basic shape checks
        # -----------------------------
        if candidates.ndim != 3:
            raise ValueError(f"candidates must be (N,K,M), got {candidates.shape}")
        if reference.ndim != 2:
            raise ValueError(f"reference must be (N,M), got {reference.shape}")
        if training_objectives.ndim != 2:
            raise ValueError(
                f"training_objectives must be (N_tr,M), got {training_objectives.shape}"
            )
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")

        n_points, k_samples, m_dims = candidates.shape
        if reference.shape != (n_points, m_dims):
            raise ValueError(
                f"reference must have shape (N,M)=({n_points},{m_dims}), got {reference.shape}"
            )
        if training_objectives.shape[1] != m_dims:
            raise ValueError(
                f"training_objectives must have M={m_dims} columns, got {training_objectives.shape[1]}"
            )

        distance = str(distance).lower().strip()

        # -----------------------------
        # 1) Residuals in normalized space (no tau)
        #    r^(k)(y*) = y_hat^(k) - y*
        # -----------------------------
        residuals = candidates - reference[:, np.newaxis, :]  # (N, K, M)

        # -----------------------------
        # 2) Discrepancy scores for ranking
        #    - Euclidean: ||r||_2
        #    - Mahalanobis: sqrt( r^T Sigma^{-1} r ), Sigma estimated from training
        # -----------------------------
        if distance == "euclidean":
            discrepancy_scores = np.linalg.norm(residuals, axis=2)  # (N, K)

        elif distance == "mahalanobis":
            # 2a) Estimate covariance in normalized space and regularize
            sigma_mat = np.cov(training_objectives, rowvar=False, bias=False)  # (M, M)
            sigma_mat = 0.5 * (sigma_mat + sigma_mat.T)  # symmetry
            sigma_mat_reg = sigma_mat + ridge * np.eye(m_dims)

            # 2b) Invert to get precision and compute quadratic form
            precision = np.linalg.inv(sigma_mat_reg)
            sq_dist = np.einsum("nkm,mj,nkj->nk", residuals, precision, residuals)
            discrepancy_scores = np.sqrt(np.maximum(sq_dist, 0.0))

        else:
            raise ValueError(f"Unknown distance: {distance!r}")

        # -----------------------------
        # 3) Ranking
        # -----------------------------
        best_shot_scores = np.min(discrepancy_scores, axis=1)  # (N,)
        rank_indices = np.argsort(discrepancy_scores, axis=1)  # (N, K)

        # -----------------------------
        # 4) Candidate-cloud diagnostics (bias and dispersion)
        #    b(y*) = ||mean_r||_2 / sqrt(M)
        #    v(y*) = median_k ||r^(k) - mean_r||_2 / sqrt(M)
        # -----------------------------
        mean_residual = np.mean(residuals, axis=1)  # (N, M)
        bias = np.linalg.norm(mean_residual, axis=1) / np.sqrt(m_dims)  # (N,)

        centered_residuals = residuals - mean_residual[:, np.newaxis, :]  # (N, K, M)
        dist_to_mean = np.linalg.norm(centered_residuals, axis=2)  # (N, K)
        dispersion = np.median(dist_to_mean, axis=1) / np.sqrt(m_dims)  # (N,)

        # -----------------------------
        # 5) Summary and spatial reporting
        # -----------------------------
        summary = AccuracySummary(
            mean_best_shot=float(np.mean(best_shot_scores)),
            median_best_shot=float(np.median(best_shot_scores)),
            mean_bias=float(np.mean(bias)),
            mean_dispersion=float(np.mean(dispersion)),
        )

        return SpatialAudit(
            residuals=residuals,
            discrepancy_scores=discrepancy_scores,
            best_shot_scores=best_shot_scores,
            rank_indices=rank_indices,
            bias=bias,
            dispersion=dispersion,
            summary=summary,
        )
