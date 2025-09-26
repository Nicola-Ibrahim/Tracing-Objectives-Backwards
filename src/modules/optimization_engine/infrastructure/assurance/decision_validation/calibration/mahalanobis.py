import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_ood_calibrator import (
    BaseOODCalibrator,
)


class MahalanobisCalibrator(BaseOODCalibrator):
    """Mahalanobis OOD: fit mean/cov on calibration decisions y, threshold by empirical percentile."""

    def __init__(self, *, percentile: float = 97.5, cov_reg: float = 1e-6) -> None:
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        if cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")
        self._percentile = float(percentile)
        self._cov_reg = float(cov_reg)
        self._mu: NDArray[np.float64] | None = None
        self._P: NDArray[np.float64] | None = None  # precision or pseudo-precision
        self._threshold: float | None = None
        self._dim: int | None = None
        self._chol: tuple[NDArray[np.float64], bool] | None = None  # (L, lower)

    def fit(self, X_cal_norm: np.ndarray) -> None:
        X = np.asarray(X_cal_norm, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X_cal_norm must be 2-D (n, d)")
        n, d = X.shape
        self._dim = d

        mu = X.mean(axis=0)
        C = np.cov(X, rowvar=False)  # (d,d)
        # regularize
        C.flat[:: d + 1] += self._cov_reg

        # Try Cholesky (fast & stable); fallback to Hermitian pseudoinverse
        chol, ok = None, True
        try:
            L = np.linalg.cholesky(C)  # C = L L^T
            chol, ok = (L, True), True
            # store precision implicitly via chol; we won’t invert explicitly
            P = None
        except np.linalg.LinAlgError:
            ok = False
            P = np.linalg.pinvh(C)  # symmetric pseudoinverse

        # md^2 for cal set
        D = X - mu
        if ok:
            # solve L z = d  -> md^2 = ||z||^2
            Z = np.linalg.solve(chol[0], D.T)  # (d,n)
            md2_vals = np.sum(Z**2, axis=0)
        else:
            md2_vals = np.einsum("ni,ij,nj->n", D, P, D)

        thr = float(np.percentile(md2_vals, self._percentile))

        self._mu = mu
        self._P = P
        self._threshold = thr
        self._chol = chol  # None if we used pinvh

    def transform(self, sample: np.ndarray) -> dict[str, NDArray | float]:
        """Return md2 for each sample and the scalar threshold."""
        if any(v is None for v in (self._mu, self._threshold)):
            raise RuntimeError("Calibrator must be fitted before transform().")
        X = np.asarray(sample, dtype=np.float64)
        X = X.reshape(1, -1) if X.ndim == 1 else X
        if self._dim is not None and X.shape[1] != self._dim:
            raise ValueError(f"Expected feature dim {self._dim}, got {X.shape[1]}")

        D = X - self._mu  # (n,d)
        if self._chol is not None:
            L = self._chol[0]
            Z = np.linalg.solve(L, D.T)  # (d,n)
            md2 = np.sum(Z**2, axis=0)  # (n,)
        else:
            if self._P is None:
                raise RuntimeError("Internal precision matrix missing.")
            md2 = np.einsum("ni,ij,nj->n", D, self._P, D)

        return {"md2": md2.astype(np.float64), "threshold": float(self._threshold)}

    def evaluate(self, sample: np.ndarray) -> tuple[bool, dict[str, float | bool], str]:
        """Evaluate the OOD gate for the provided sample."""
        metrics = self.transform(sample)
        try:
            md2_raw = metrics["md2"]
            threshold_md2 = float(metrics["threshold"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                "OOD calibrator must supply 'md2' and 'threshold' keys in transform()."
            ) from exc

        md2 = float(np.asarray(md2_raw, dtype=float).max())
        passed = md2 <= threshold_md2
        gate_metrics: dict[str, float | bool] = {
            "gate1_md2": md2,
            "gate1_md2_threshold": threshold_md2,
            "gate1_inlier": bool(passed),
        }
        explanation = (
            "Pass: candidate is within the supported decision region."
            if passed
            else "ABSTAIN: candidate is outside the supported decision region."
        )
        return bool(passed), gate_metrics, explanation

    # ---- properties ----
    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Calibrator not fitted.")
        return self._threshold

    @property
    def mu(self) -> NDArray[np.float64]:
        if self._mu is None:
            raise RuntimeError("Calibrator not fitted.")
        return self._mu

    @property
    def precision(self) -> NDArray[np.float64] | None:
        # May be None if we used Cholesky path
        return self._P

    @property
    def cov_reg(self) -> float:
        return self._cov_reg

    @property
    def percentile(self) -> float:
        return self._percentile
