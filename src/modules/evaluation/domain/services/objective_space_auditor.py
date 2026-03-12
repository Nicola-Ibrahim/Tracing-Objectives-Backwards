import numpy as np
from ..value_objects.objective_assessment import ObjectiveSpaceAssessment
from ..value_objects.ecdf_profile import ECDFProfile
from typing import Literal

DistanceType = Literal["euclidean", "mahalanobis"]

class ObjectiveSpaceAuditor:
    """
    Domain Service: Assessment in *normalized* objective space.
    Provides ObjectiveSpaceAssessment including ECDF of error distributions.
    """

    @staticmethod
    def audit(
        training_objectives: np.ndarray,
        candidates: np.ndarray,
        reference: np.ndarray,
        distance: DistanceType = "euclidean",
        ridge: float = 1e-6,
    ) -> ObjectiveSpaceAssessment:
        """
        Calculates accuracy metrics in objective space.

        Args:
            training_objectives: (N_tr, M) training outcomes in normalized space.
            candidates: (N, K, M) candidate outcomes (y_hat) in normalized space.
            reference: (N, M) target outcomes (y*) in normalized space.
            distance: "euclidean" | "mahalanobis".
        """
        n_points, k_samples, m_dims = candidates.shape
        
        # 1. Residuals (y_hat - y*)
        residuals = candidates - reference[:, np.newaxis, :]  # (N, K, M)

        # 2. Discrepancy scores
        if distance == "euclidean":
            discrepancy_scores = np.linalg.norm(residuals, axis=2)  # (N, K)
        elif distance == "mahalanobis":
            sigma_mat = np.cov(training_objectives, rowvar=False, bias=False)
            sigma_mat = 0.5 * (sigma_mat + sigma_mat.T)
            sigma_mat_reg = sigma_mat + ridge * np.eye(m_dims)
            precision = np.linalg.inv(sigma_mat_reg)
            sq_dist = np.einsum("nkm,mj,nkj->nk", residuals, precision, residuals)
            discrepancy_scores = np.sqrt(np.maximum(sq_dist, 0.0))
        else:
            raise ValueError(f"Unknown distance: {distance}")

        # 3. Best shot scores (closest per target)
        best_shot_scores = np.min(discrepancy_scores, axis=1)  # (N,)

        # 4. Bias and Dispersion
        mean_residual = np.mean(residuals, axis=1)  # (N, M)
        bias = np.linalg.norm(mean_residual, axis=1) / np.sqrt(m_dims)  # (N,)
        
        centered_residuals = residuals - mean_residual[:, np.newaxis, :]
        dist_to_mean = np.linalg.norm(centered_residuals, axis=2)
        dispersion = np.median(dist_to_mean, axis=1) / np.sqrt(m_dims)  # (N,)

        # 5. ECDF Profile of best_shot_scores
        ecdf_profile = ObjectiveSpaceAuditor._compute_ecdf(best_shot_scores)

        return ObjectiveSpaceAssessment(
            ecdf_profile=ecdf_profile,
            mean_best_shot=float(np.mean(best_shot_scores)),
            median_best_shot=float(np.median(best_shot_scores)),
            mean_bias=float(np.mean(bias)),
            mean_dispersion=float(np.mean(dispersion)),
        )

    @staticmethod
    def _compute_ecdf(scores: np.ndarray, max_points: int = 100) -> ECDFProfile:
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        
        if n > max_points:
            indices = np.linspace(0, n - 1, max_points, dtype=int)
            x_values = sorted_scores[indices].tolist()
            y_values = ((indices + 1) / n).tolist()
        else:
            x_values = sorted_scores.tolist()
            y_values = (np.arange(1, n + 1) / n).tolist()
            
        return ECDFProfile(
            x_values=x_values,
            cumulative_probabilities=y_values
        )
