import numpy as np

from ..forward_models.forward_ensemble import ForwardEnsemble


class ConformalSplitL2:
    def __init__(self, confidence: float = 0.90):
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0, 1)")
        self._conf = confidence
        self._q: float | None = None

    def fit(
        self,
        X_cal_norm: np.ndarray,
        Y_cal_norm: np.ndarray,
        ensemble: ForwardEnsemble,
    ) -> None:
        y_hat = ensemble.predict_mean(X_cal_norm)
        residuals = np.linalg.norm(Y_cal_norm - y_hat, axis=1)
        self._q = float(np.quantile(residuals, self._conf))

    def predictive_radius(self) -> float:
        if self._q is None:
            raise RuntimeError("ConformalSplitL2 must be fit before use.")
        return self._q

    def is_within_tolerance(
        self,
        *,
        y_hat_norm: np.ndarray,
        y_star_norm: np.ndarray,
        eps_l2: float | None,
        eps_per_obj: np.ndarray | None,
    ) -> bool:
        if self._q is None:
            raise RuntimeError("ConformalSplitL2 must be fit before use.")
        diff = np.abs(y_hat_norm - y_star_norm)
        if eps_l2 is not None:
            return float(np.linalg.norm(diff) + self._q) <= eps_l2
        if eps_per_obj is not None:
            return bool(np.all(diff + self._q <= eps_per_obj))
        raise ValueError("Provide eps_l2 or eps_per_obj tolerances.")


__all__ = ["ConformalSplitL2"]
