import numpy as np

from .calibration import OODCalibration


def calibrate_mahalanobis(
    X_cal_norm: np.ndarray, *, percentile: float = 97.5, cov_reg: float = 1e-6
) -> OODCalibration:
    if X_cal_norm.ndim != 2:
        raise ValueError("X_cal_norm must be a 2-D array")
    mu = X_cal_norm.mean(axis=0)
    cov = np.cov(X_cal_norm, rowvar=False) + cov_reg * np.eye(X_cal_norm.shape[1])
    prec = np.linalg.inv(cov)
    d = X_cal_norm - mu
    md2_vals = np.einsum("ni,ij,nj->n", d, prec, d)
    threshold = float(np.percentile(md2_vals, percentile))
    return OODCalibration(mu=mu, prec=prec, threshold_md2=threshold)


__all__ = ["calibrate_mahalanobis"]
