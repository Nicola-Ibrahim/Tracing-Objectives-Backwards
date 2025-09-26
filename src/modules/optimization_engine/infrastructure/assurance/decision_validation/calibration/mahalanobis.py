import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces import OODCalibrator


class MahalanobisCalibrator(OODCalibrator):
    """Mahalanobis distance-based OOD calibrator."""

    def __init__(self, *, percentile: float = 97.5, cov_reg: float = 1e-6) -> None:
        """Initialize the calibrator with the desired percentile and covariance regularization.
        Args:
            percentile (float, optional): Percentile for the Mahalanobis distance threshold.
                Must be in (0, 100]. Defaults to 97.5.
            cov_reg (float, optional): Regularization term added to the diagonal of the
                covariance matrix to ensure numerical stability. Must be non-negative.
                Defaults to 1e-6.
        """
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        if cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")
        self._percentile = percentile
        self._cov_reg = cov_reg
        self._mu: NDArray[np.float64] | None = None
        self._prec: NDArray[np.float64] | None = None
        self._threshold: float | None = None

    def fit(self, X_cal_norm: np.ndarray) -> None:
        X_cal_norm = np.asarray(X_cal_norm, dtype=float)
        if X_cal_norm.ndim != 2:
            raise ValueError("X_cal_norm must be a 2-D array")
        mu = X_cal_norm.mean(axis=0)
        cov = np.cov(X_cal_norm, rowvar=False) + self._cov_reg * np.eye(
            X_cal_norm.shape[1]
        )
        prec = np.linalg.inv(cov)
        d = X_cal_norm - mu
        md2_vals = np.einsum("ni,ij,nj->n", d, prec, d)
        threshold = float(np.percentile(md2_vals, self._percentile))

        self._mu = np.asarray(mu, dtype=float)
        self._prec = np.asarray(prec, dtype=float)
        self._threshold = threshold

    def transform(self, sample: np.ndarray) -> dict[str, float]:
        if self._mu is None or self._prec is None or self._threshold is None:
            raise RuntimeError(
                "MahalanobisCalibrator must be fitted before calling transform()."
            )
        s = np.atleast_2d(np.asarray(sample, dtype=float))
        delta = s - self._mu
        md2 = float(np.einsum("ni,ij,nj->n", delta, self._prec, delta)[0])
        return {"md2": md2, "threshold": self._threshold}

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("MahalanobisCalibrator has not been fitted yet.")
        return self._threshold
