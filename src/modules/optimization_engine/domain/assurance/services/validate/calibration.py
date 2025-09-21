# domain/assurance/services/calibration.py
from __future__ import annotations

import numpy as np

from ...modeling.services.forward_ensemble import ForwardEnsemble
from ..value_objects import ConformalCalibration, OODCalibration


def calibrate_mahalanobis(
    X_cal_norm: np.ndarray, *, percentile: float = 97.5, cov_reg: float = 1e-6
) -> OODCalibration:
    """
    Fit mean/cov on NORMALISED X and compute empirical MD^2 threshold.
    """
    if X_cal_norm.ndim != 2:
        raise ValueError("X_cal_norm must be (N, d_x)")
    mu = X_cal_norm.mean(axis=0)
    cov = np.cov(X_cal_norm, rowvar=False) + cov_reg * np.eye(X_cal_norm.shape[1])
    prec = np.linalg.inv(cov)
    d = X_cal_norm - mu
    md2_vals = np.einsum("ni,ij,nj->n", d, prec, d)
    thr = float(np.percentile(md2_vals, percentile))
    return OODCalibration(mu=mu, prec=prec, threshold_md2=thr)


def calibrate_split_conformal_l2(
    X_cal_norm: np.ndarray,
    Y_cal_norm: np.ndarray,
    ensemble: ForwardEnsemble,
    *,
    confidence: float = 0.90,
) -> ConformalCalibration:
    """
    Compute q as the confidence-quantile of joint-L2 residual norms on calibration split.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")
    y_hat = ensemble.predict_mean(X_cal_norm)  # (N, d_y)
    r = np.linalg.norm(Y_cal_norm - y_hat, axis=1)  # (N,)
    q = float(np.quantile(r, confidence))
    return ConformalCalibration(radius_q=q)
