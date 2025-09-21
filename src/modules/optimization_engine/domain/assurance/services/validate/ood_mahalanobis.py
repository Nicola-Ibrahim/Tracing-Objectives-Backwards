import numpy as np


class OODMahal:
    """
    Mahalanobis OOD detector in NORMALISED X space.

    - Fit on calibration X_norm to get (mu, cov^-1) and an empirical MD^2 threshold.
    - Score a single x_norm and decide inlier vs. outlier.

    Note: this is intentionally simple and transparent.
    """

    def __init__(self, cov_reg: float = 1e-6):
        self._mu: np.ndarray | None = None
        self._prec: np.ndarray | None = None
        self._thr: float | None = None
        self._cov_reg = cov_reg

    @staticmethod
    def _md2(x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
        d = x - mu
        return np.einsum("...i,ij,...j->...", d, prec, d)

    def fit(self, X_cal_norm: np.ndarray, percentile: float) -> None:
        """
        Parameters
        ----------
        X_cal_norm : (N, d_x) normalised calibration inputs
        percentile : float
            Empirical percentile for MD^2 threshold (e.g., 97.5).
        """
        if X_cal_norm.ndim != 2:
            raise ValueError("X_cal_norm must be 2D.")
        self._mu = X_cal_norm.mean(axis=0)
        cov = np.cov(X_cal_norm, rowvar=False)
        cov = cov + self._cov_reg * np.eye(cov.shape[0])
        self._prec = np.linalg.inv(cov)
        md2_vals = self._md2(X_cal_norm, self._mu, self._prec)
        self._thr = float(np.percentile(md2_vals, percentile))

    def score(self, x_norm: np.ndarray) -> float:
        if self._mu is None or self._prec is None:
            raise RuntimeError("OODMahal must be fit() before score().")
        x_norm = np.atleast_2d(x_norm)
        return float(self._md2(x_norm, self._mu, self._prec)[0])

    def is_inlier(self, x_norm: np.ndarray) -> bool:
        if self._thr is None:
            raise RuntimeError("OODMahal has no threshold; call fit() first.")
        return self.score(x_norm) <= self._thr

    @property
    def threshold(self) -> float:
        if self._thr is None:
            raise RuntimeError("Threshold not set; call fit().")
        return self._thr
