import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalCalibrator,
)


class SplitConformalL2Calibrator(BaseConformalCalibrator):
    """
    Split-conformal calibrator for the L2 error of a *forward* mapper f_hat: X -> y.

    Assumptions
    -----------
    - The attached estimator implements an inverse map: inputs are objectives y (R^{d_y}),
      outputs are decisions X (R^{d_x}), i.e., f_hat(y) ≈ X.
    - Calibration data are paired decisions and objectives (X_i, y_i).

    API (agnostic but consistent)
    -----------------------------
    fit(X, y):            trains estimator on (y -> X) and learns a global L2 "cushion" radius_q
    evaluate(y, X_target, tolerance): checks if f_hat(y) is within tolerance of X_target after adding radius_q

    Coverage intuition
    ------------------
    radius_q is the (finite-sample corrected) high quantile of past residual norms,
    giving distribution-free guarantees under exchangeability. See Angelopoulos & Bates (2023),
    Tibshirani (2024), Romano et al. (2019).  # refs in chat
    """

    def __init__(self, estimator, confidence: float = 0.9) -> None:
        super().__init__(estimator=estimator, confidence=confidence)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")
        self._confidence: float = float(confidence)
        self._radius_q: float | None = None
        self._n_cal: int | None = None
        self._x_dim: int | None = None
        self._y_dim: int | None = None

    # ----------------------------- calibration ----------------------------- #
    def fit(self, X: NDArray, y: NDArray) -> None:
        """
        Learn f_hat on (X -> y) and compute the split-conformal radius_q from L2 residuals.

        Steps:
          1) estimator.fit(X, y)
          2) residuals r_i = || estimator.predict(X_i) - y_i ||_2
          3) radius_q = quantile_tau(residuals), tau = ceil((n+1)*confidence)/n
        """
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D arrays (n, d)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n, d_x = X.shape
        _, d_y = y.shape
        self._x_dim, self._y_dim = d_x, d_y

        # Train forward map: inputs=X, targets=y
        self.estimator.fit(X, y)

        y_hat = self.estimator.predict(X)  # (n, d_y)
        if y_hat.shape != y.shape:
            raise ValueError(f"predict returned {y_hat.shape}, expected {y.shape}")

        residuals = np.linalg.norm(y_hat - y, axis=1)  # (n,)
        self._n_cal = int(residuals.size)

        # Finite-sample split-conformal quantile (conservative rounding). :contentReference[oaicite:2]{index=2}
        tau = float(np.ceil((self._n_cal + 1) * self._confidence) / self._n_cal)
        tau = min(max(tau, 0.0), 1.0)
        self._radius_q = float(np.quantile(residuals, tau, method="higher"))

    # ----------------------------- evaluation ----------------------------- #
    def evaluate(
        self,
        X: NDArray[np.float64],
        y_target: NDArray[np.float64],
        tolerance: float,
    ) -> tuple[bool, dict[str, float | bool], str]:
        """
        Check whether f_hat(X) is acceptable for y_target under a global L2 tolerance,
        after adding the conformal cushion radius_q.

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

        diag = self._predict_and_distances(X=X, y_target=y_target)
        y_hat = diag["y_hat"]
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
    def _predict_and_distances(
        self, *, X: NDArray[np.float64], y_target: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64] | float]:
        """
        Predict outcomes for X, and compute distances to y_target.

        Returns:
            {
              "y_hat":       (n, d_y) predicted outcomes,
              "l2_distance": float (max L2 across batch)
            }
        """
        if self._radius_q is None:
            raise RuntimeError("Calibrator must be fitted before evaluation.")

        # enforce (n, d)
        if X.ndim == 1:
            X = X[None, :]
        if y_target.ndim == 1:
            y_target = y_target[None, :]

        # shape checks
        if self._x_dim is not None and X.shape[1] != self._x_dim:
            raise ValueError(f"X has dim {X.shape[1]}, expected {self._x_dim}")
        if self._y_dim is not None and y_target.shape[1] != self._y_dim:
            raise ValueError(
                f"y_target has dim {y_target.shape[1]}, expected {self._y_dim}"
            )
        if X.shape[0] != y_target.shape[0]:
            raise ValueError("Batch sizes must match between X and y_target")

        y_hat = self.estimator.predict(X)  # (n, d_y)
        if y_hat.shape != y_target.shape:
            raise ValueError(
                f"predict returned {y_hat.shape}, expected {y_target.shape}"
            )

        l2_all = np.linalg.norm(y_hat - y_target, axis=1)  # (n,)
        l2_distance = float(l2_all.max())  # conservative batch summary

        return {
            "y_hat": y_hat.astype(np.float64, copy=False),
            "l2_distance": l2_distance,
        }

    # ----------------------------- properties ----------------------------- #
    @property
    def radius_q(self) -> float:
        if self._radius_q is None:
            raise RuntimeError("Calibrator must be fitted before accessing radius_q.")
        return float(self._radius_q)

    @property
    def confidence(self) -> float:
        return self._confidence
