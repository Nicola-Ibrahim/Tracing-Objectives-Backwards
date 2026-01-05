import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_ood_calibrator import (
    BaseOODValidator,
)


class MahalanobisOODValidator(BaseOODValidator):
    """
    Mahalanobis-based OOD calibrator.

    Intuition
    ---------
    We want a simple, data-driven test for whether a new sample looks like the
    calibration data (i.e., is "in-distribution"). We estimate the calibration
    cloud's center and spread, then score new samples by how far they are from
    that cloud using Mahalanobis distance:
        md²(x) = (x - center)ᵀ Σ⁻¹ (x - center)
    Small md² ⇒ close to the cloud (inlier); large md² ⇒ far (outlier).
    We set an **inlier_threshold** as an empirical percentile of md² over the
    calibration set.

    What fit() does
    ---------------
    1) Estimate the **center** (mean vector) and a **regularized covariance**
       Σ̂ = Cov(X) + λI for numerical stability (λ = covariance_ridge).
    2) Prefer a Cholesky factorization L where Σ̂ = L Lᵀ (stable/fast). If Σ̂ is
       not SPD, fall back to the Hermitian pseudoinverse Σ̂⁺.
    3) Compute md² for all calibration rows and set **inlier_threshold** to the
       chosen empirical percentile (e.g., 97.5th). This defines the inlier ellipsoid.

    What evaluate(...) does
    -----------------------
    1) For a new sample (or batch), compute md² using the fitted center and Σ̂.
    2) Summarize a batch conservatively by the maximum md² in the batch.
    3) Return pass/fail based on md² ≤ inlier_threshold, plus readable metrics
       and a short explanation string.
    """

    def __init__(self, *, percentile: float = 97.5, cov_reg: float = 1e-6) -> None:
        """
        Args:
            percentile: Empirical percentile (in (0, 100]) used as the md² cutoff.
                        Example: 97.5 keeps ~97.5% of calibration points as inliers.
            cov_reg   : Non-negative diagonal regularization λ added to Σ (ridge).

        Raises:
            ValueError: if percentile is out of (0,100] or cov_reg < 0.
        """
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        if cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")

        # Policy / hyperparameters
        self._inlier_percentile = float(percentile)
        self._covariance_ridge = float(cov_reg)

        # Fitted state
        self._center: NDArray[np.float64] | None = None  # μ
        self._inverse_covariance: NDArray[np.float64] | None = (
            None  # Σ̂⁺ (may be None if Cholesky is used)
        )
        self._inlier_threshold: float | None = None  # cutoff on md²
        self._feature_dim: int | None = None  # d
        self._chol_factor: NDArray[np.float64] | None = None  # L s.t. Σ̂ = L Lᵀ (if SPD)

    # --------------------------------------------------------------------- #
    # Calibration
    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the inlier region on calibration data.

        Steps:
          1) Compute center (mean) and regularized covariance Σ̂ = Cov(X) + λI.
          2) Prefer Cholesky (Σ̂ = L Lᵀ). Otherwise store Σ̂⁺ via pinvh(Σ̂).
          3) Compute md² for all calibration rows; set inlier_threshold to the
             configured percentile of md².

        Args:
            X: Calibration matrix of shape (n, d).

        Raises:
            ValueError: if X is not 2-D.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, d)")

        n, d = X.shape
        self._feature_dim = d

        # Center and (regularized) covariance
        center = X.mean(axis=0)  # (d,)
        covariance = np.cov(X, rowvar=False)  # (d,d)
        covariance.flat[:: d + 1] += self._covariance_ridge  # add λ to diagonal

        # Factorization / inverse-covariance strategy
        chol_factor = None
        inverse_covariance = None
        try:
            chol_factor = np.linalg.cholesky(covariance)  # Σ̂ = L Lᵀ
        except np.linalg.LinAlgError:
            chol_factor = None
            inverse_covariance = np.linalg.pinvh(
                covariance
            )  # Σ̂⁺, symmetric pseudoinverse

        # md² over calibration set
        centered = X - center  # (n,d)
        if chol_factor is not None:
            # Solve L z = (x - μ)^T; md² = ||z||²
            z = np.linalg.solve(chol_factor, centered.T)  # (d,n)
            mahalanobis_sq = np.sum(z**2, axis=0)  # (n,)
        else:
            mahalanobis_sq = np.einsum(
                "ni,ij,nj->n", centered, inverse_covariance, centered
            )

        # Empirical percentile threshold
        inlier_threshold = float(np.percentile(mahalanobis_sq, self._inlier_percentile))

        # Persist fitted state
        self._center = center
        self._inverse_covariance = inverse_covariance
        self._inlier_threshold = inlier_threshold
        self._chol_factor = chol_factor

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #
    def _compute_diagnostics(
        self, X: np.ndarray
    ) -> dict[str, float | NDArray[np.float64]]:
        """
        Compute md² for one or more samples and package diagnostics.

        Args:
            X: Array of shape (d,) or (m, d).

        Returns:
            {
              "mahalanobis_sq": (m,) vector of md² values,
              "inlier_threshold": float, learned md² cutoff
            }

        Raises:
            RuntimeError: if not fitted.
            ValueError:   if feature dimension mismatches.
        """
        if self._center is None or self._inlier_threshold is None:
            raise RuntimeError("Calibrator must be fitted before evaluation.")

        X = np.asarray(X, dtype=np.float64)
        X = X.reshape(1, -1) if X.ndim == 1 else X
        if self._feature_dim is not None and X.shape[1] != self._feature_dim:
            raise ValueError(
                f"Expected feature dim {self._feature_dim}, got {X.shape[1]}"
            )

        centered = X - self._center  # (m,d)
        if self._chol_factor is not None:
            z = np.linalg.solve(self._chol_factor, centered.T)  # (d,m)
            md2 = np.sum(z**2, axis=0)  # (m,)
        else:
            if self._inverse_covariance is None:
                raise RuntimeError("Internal inverse covariance is missing.")
            md2 = np.einsum("ni,ij,nj->n", centered, self._inverse_covariance, centered)

        return {
            "mahalanobis_sq": md2.astype(np.float64),
            "inlier_threshold": float(self._inlier_threshold),
        }

    def validate(self, X: np.ndarray) -> tuple[bool, dict[str, float | bool], str]:
        """
        Decide inlier/outlier for a sample (or batch) using the learned cutoff.

        Rule:
            inlier  ⇔  max(md²(samples)) <= inlier_threshold
            (Batch is summarized conservatively by the maximum md².)

        Returns:
            passed: bool (True if inlier)
            metrics: {
               "mahalanobis_sq": float, max md² over the batch,
               "inlier_threshold": float, learned cutoff,
               "inlier": bool,

               # Backward-compatibility (original keys):
               "md2": float,
               "md2_threshold": float
            }
            explanation: short human-readable message.
        """
        diag = self._compute_diagnostics(X)
        md2_vec = np.asarray(diag["mahalanobis_sq"], dtype=float)  # (m,)
        inlier_threshold = float(diag["inlier_threshold"])

        md2_max = float(md2_vec.max())
        passed = md2_max <= inlier_threshold

        metrics: dict[str, float | bool] = {
            "mahalanobis_sq": md2_max,
            "inlier_threshold": inlier_threshold,
            "inlier": bool(passed),
            # compatibility with previous naming
            "md2": md2_max,
            "md2_threshold": inlier_threshold,
        }
        explanation = (
            "Pass: candidate is within the supported region (Mahalanobis inlier)."
            if passed
            else "ABSTAIN: candidate lies outside the supported region (Mahalanobis outlier)."
        )
        return bool(passed), metrics, explanation
