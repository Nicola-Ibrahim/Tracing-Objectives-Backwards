"""Infrastructure OOD calibration using Mahalanobis distance."""

import numpy as np

from .....domain.assurance.decision_validation.interfaces import OODCalibrator
from .....domain.assurance.decision_validation.value_objects.calibration import (
    OODCalibration,
)


class MahalanobisCalibrator(OODCalibrator):
    def __init__(self, *, percentile: float = 97.5, cov_reg: float = 1e-6) -> None:
        if not (0 < percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        if cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")
        self._percentile = percentile
        self._cov_reg = cov_reg

    def fit(self, X_cal_norm: np.ndarray) -> OODCalibration:
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
        return OODCalibration(mu=mu, prec=prec, threshold_md2=threshold)


__all__ = ["MahalanobisCalibrator"]
