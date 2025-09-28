import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalCalibrator,
)


class SplitConformalL2Calibrator(BaseConformalCalibrator):
    """
    Split-conformal calibrator for the L2 error of an inverse mapper f_hat: Y -> X.

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
        """
        Args:
            estimator: inverse mapper expecting .fit(inputs=y, targets=X) and .predict(y)->X_hat
            confidence: desired marginal coverage level in (0, 1) (e.g., 0.9 or 0.95)
        """
        super().__init__(estimator)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")
        self._confidence: float = float(confidence)
        self._radius_q: float | None = None
        self._n_cal: int | None = None
        self._x_dim: int | None = None
        self._y_dim: int | None = None

    # ----------------------------- calibration ----------------------------- #
    def fit(
        self,
        X: NDArray[np.float64],  # (n, d_x) decisions
        y: NDArray[np.float64],  # (n, d_y) objectives
    ) -> None:
        """
        Learn f_hat on (y -> X) and compute the split-conformal radius_q from residual L2 norms.

        Steps:
          1) estimator.fit(y, X)
          2) residuals r_i = || estimator.predict(y_i) - X_i ||_2
          3) radius_q = quantile_tau(residuals), tau = ceil((n+1)*confidence)/n (finite-sample)

        Raises:
            ValueError: on shape mismatches or unexpected estimator output.
        """
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D arrays (n, d)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n, d_x = X.shape
        _, d_y = y.shape
        self._x_dim, self._y_dim = d_x, d_y

        # Train inverse map: inputs=y, targets=X
        self.estimator.fit(y, X)

        X_hat = self.estimator.predict(y)  # (n, d_x)
        if X_hat.shape != X.shape:
            raise ValueError(f"predict returned {X_hat.shape}, expected {X.shape}")

        residuals = np.linalg.norm(X_hat - X, axis=1)  # (n,)
        self._n_cal = int(residuals.size)

        # Finite-sample split-conformal quantile for coverage ~= confidence
        tau = float(np.ceil((self._n_cal + 1) * self._confidence) / self._n_cal)
        tau = min(max(tau, 0.0), 1.0)

        # Use "higher" to ensure conservative (>=) coverage
        self._radius_q = float(np.quantile(residuals, tau, method="higher"))

    # ----------------------------- evaluation ----------------------------- #
    def evaluate(
        self,
        *,
        y: NDArray[np.float64],  # candidate objectives (n, d_y) or (d_y,)
        X_target: NDArray[np.float64],  # target decisions   (n, d_x) or (d_x,)
        tolerance: float,
    ) -> tuple[bool, dict[str, float | bool], str]:
        """
        Check whether f_hat(y) is acceptable for X_target under a global L2 tolerance,
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

        diag = self._predict_and_distances(X_target=X_target, y=y)
        X_hat = diag["X_hat"]
        l2_distance = diag["l2_distance"]

        radius_q = self.radius_q  # learned cushion
        passed = (l2_distance + radius_q) <= tolerance

        metrics: dict[str, float | bool] = {
            "conformal_radius_q": radius_q,
            "dist_to_target_l2": l2_distance,
            "covered": bool(passed),
        }
        explanation = (
            "PASS: predicted decisions within tolerance after adding conformal cushion."
            if passed
            else "ABSTAIN: predicted decisions exceed tolerance after adding conformal cushion."
        )
        return bool(passed), metrics, explanation

    # ----------------------------- helpers ----------------------------- #
    def _predict_and_distances(
        self, *, X_target: NDArray[np.float64], y: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64] | float]:
        """
        Predict decisions for y, and compute distances to X_target.

        Returns:
            {
              "X_hat":       (n, d_x) predicted decisions,
              "l2_distance": float (max L2 across batch)
            }
        """
        if self._radius_q is None:
            raise RuntimeError("Calibrator must be fitted before evaluation.")

        # enforce (n, d)
        if y.ndim == 1:
            y = y[None, :]
        if X_target.ndim == 1:
            X_target = X_target[None, :]

        # shape checks
        if self._y_dim is not None and y.shape[1] != self._y_dim:
            raise ValueError(f"y has dim {y.shape[1]}, expected {self._y_dim}")
        if self._x_dim is not None and X_target.shape[1] != self._x_dim:
            raise ValueError(
                f"X_target has dim {X_target.shape[1]}, expected {self._x_dim}"
            )
        if y.shape[0] != X_target.shape[0]:
            raise ValueError("Batch sizes must match between y and X_target")

        X_hat = self.estimator.predict(y)  # (n, d_x)
        if X_hat.shape != X_target.shape:
            raise ValueError(
                f"predict returned {X_hat.shape}, expected {X_target.shape}"
            )

        l2_all = np.linalg.norm(X_hat - X_target, axis=1)  # (n,)
        l2_distance = float(l2_all.max())  # conservative batch summary

        return {
            "X_hat": X_hat.astype(np.float64, copy=False),
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
