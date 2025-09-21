import numpy as np

from .forward_ensemble import ForwardEnsemble


class ConformalSplitL2:
    """
    Split-conformal calibrator for NORMALISED Y using a JOINT L2 residual norm.

    Produces a scalar radius q such that the L2 ball around y_hat has ~`confidence` coverage.
    """

    def __init__(self, confidence: float = 0.90):
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0,1).")
        self._conf = confidence
        self._q: float | None = None

    def fit(
        self, X_cal_norm: np.ndarray, Y_cal_norm: np.ndarray, ensemble: ForwardEnsemble
    ) -> None:
        """
        Compute residual L2 norms on a held-out calibration split and store q_{confidence}.
        """
        y_hat = ensemble.predict_mean(X_cal_norm)  # (N, d_y)
        r = np.linalg.norm(Y_cal_norm - y_hat, axis=1)  # (N,)
        self._q = float(np.quantile(r, self._conf))

    def predictive_radius(self) -> float:
        if self._q is None:
            raise RuntimeError("ConformalSplitL2 not fit.")
        return self._q

    def is_within_tolerance(
        self,
        y_hat_norm: np.ndarray,
        y_star_norm: np.ndarray,
        eps_l2: float | None,
        eps_per_obj: np.ndarray | None,
    ) -> bool:
        """
        Strict rule: predictive set around y_hat (L2 ball of radius q) must fit within
        the tolerance set around y*.

        If using L2 tolerance: ||y_hat - y*||_2 + q <= eps_l2
        If using per-objective: |y_hat_i - y*_i| + q <= eps_i, all i (conservative).
        """
        if self._q is None:
            raise RuntimeError("ConformalSplitL2 not fit.")
        diff = np.abs(y_hat_norm - y_star_norm)
        if eps_l2 is not None:
            return float(np.linalg.norm(diff) + self._q) <= eps_l2
        if eps_per_obj is not None:
            return np.all(diff + self._q <= eps_per_obj)
        raise ValueError("Provide either eps_l2 or eps_per_obj.")
