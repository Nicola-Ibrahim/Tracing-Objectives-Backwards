import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalValidator,
)


class SplitConformalL2Validator(BaseConformalValidator):
    """
    Split-conformal validator for the L2 error of *already predicted* objectives.

    Assumptions
    -----------
    - Calibration data are paired objective predictions and objective ground-truth
      (y_pred_i, y_true_i).

    API (agnostic but consistent)
    -----------------------------
    fit(y_pred, y_true):            learns a global L2 "cushion" radius_q from calibration residuals
    evaluate(y_pred, y_target, tolerance): checks if y_pred is within tolerance of y_target after adding radius_q

    Coverage intuition
    ------------------
    radius_q is the (finite-sample corrected) high quantile of past residual norms,
    giving distribution-free guarantees under exchangeability. See Angelopoulos & Bates (2023),
    Tibshirani (2024), Romano et al. (2019).  # refs in chat
    """

    def __init__(self, confidence: float = 0.9, **_kwargs) -> None:
        super().__init__(confidence=confidence)
        self._confidence: float = float(confidence)
        self._radius_q: float | None = None
        self._n_cal: int | None = None
        self._y_dim: int | None = None

    # ----------------------------- calibration ----------------------------- #
    def fit(self, y_pred: NDArray, y_true: NDArray) -> None:
        """
        Compute the split-conformal radius_q from L2 residuals ||y_pred - y_true||_2.

        Steps:
          1) residuals r_i = || y_pred_i - y_true_i ||_2
          2) radius_q = quantile_tau(residuals), tau = ceil((n+1)*confidence)/n
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        if y_pred.ndim != 2 or y_true.ndim != 2:
            raise ValueError("y_pred and y_true must be 2D arrays (n, d_y)")
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        n, d_y = y_true.shape
        self._y_dim = d_y

        residuals = np.linalg.norm(y_pred - y_true, axis=1)  # (n,)
        self._n_cal = int(residuals.size)

        # Finite-sample split-conformal quantile (conservative rounding).
        tau = float(np.ceil((self._n_cal + 1) * self._confidence) / self._n_cal)
        tau = min(max(tau, 0.0), 1.0)
        self._radius_q = float(np.quantile(residuals, tau, method="higher"))

    # ----------------------------- evaluation ----------------------------- #
    def validate(
        self,
        y_pred: NDArray[np.float64],
        y_target: NDArray[np.float64],
        tolerance: float,
    ) -> tuple[bool, dict[str, float | bool], str]:
        """
        Check whether y_pred is acceptable for y_target under a global L2 tolerance,
        after adding the conformal cushion radius_q (learned from calibration).

        Returns:
            passed: bool
            metrics: {
                "conformal_radius_q": float,
                "dist_to_target_l2":  float (max across batch),
                "covered":            bool
            }
            explanation: string
        """
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")

        diag = self._compute_distance_to_target(y_pred=y_pred, y_target=y_target)
        l2_distance = diag["l2_distance"]

        radius_q = self.radius_q
        passed = (l2_distance + radius_q) <= tolerance

        metrics: dict[str, float | bool] = {
            "conformal_radius_q": radius_q,
            "dist_to_target_l2": l2_distance,
            "covered": bool(passed),
        }
        explanation = (
            "PASS: predicted outcomes within tolerance after adding conformal cushion."
            if passed
            else "ABSTAIN: predicted outcomes exceed tolerance after adding conformal cushion."
        )
        return bool(passed), metrics, explanation

    # ----------------------------- helpers ----------------------------- #
    def _compute_distance_to_target(
        self, *, y_pred: NDArray[np.float64], y_target: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64] | float]:
        """
        Compute distances from predictions to target.

        Returns:
            {
              "l2_distance": float (max L2 across batch)
            }
        """
        if self._radius_q is None:
            raise RuntimeError("Calibrator must be fitted before evaluation.")

        # enforce (n, d)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        if y_target.ndim == 1:
            y_target = y_target[None, :]

        # shape checks
        if self._y_dim is not None and y_target.shape[1] != self._y_dim:
            raise ValueError(
                f"y_target has dim {y_target.shape[1]}, expected {self._y_dim}"
            )
        if y_pred.shape != y_target.shape:
            raise ValueError("y_pred and y_target must have the same shape")

        l2_all = np.linalg.norm(y_pred - y_target, axis=1)  # (n,)
        l2_distance = float(l2_all.max())  # conservative batch summary

        return {
            "l2_distance": l2_distance,
        }

    # ----------------------------- properties ----------------------------- #
    @property
    def radius_q(self) -> float:
        if self._radius_q is None:
            raise RuntimeError("Validator must be fitted before accessing radius_q.")
        return float(self._radius_q)

    @property
    def confidence(self) -> float:
        return self._confidence
