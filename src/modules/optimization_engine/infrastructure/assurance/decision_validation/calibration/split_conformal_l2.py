from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .....domain.assurance.decision_validation.interfaces.base_conformal_calibrator import (
    BaseConformalCalibrator,
)
from .....domain.modeling.interfaces.base_estimator import BaseEstimator


class SplitConformalL2Calibrator(BaseConformalCalibrator):
    """
    Split conformal calibration for L2 error of a forward mapper f_hat: y -> x.

    API:
      - fit(Y_cal_norm, X_cal_norm)
          Fits the attached estimator on (Y -> X) and learns radius_q from
          calibration residuals ||f_hat(y_i) - x_i||_2.
      - evaluate(candidate, target, eps_l2, eps_per_obj)
          Returns a gate result using the fitted conformal radius and tolerance band.
    """

    def __init__(self, *, estimator: BaseEstimator, confidence: float = 0.90) -> None:
        super().__init__(estimator=estimator)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0,1)")
        self._confidence = float(confidence)
        self._radius: float | None = None
        self._n_cal: int = 0
        self._dx: int | None = None
        self._dy: int | None = None

    # ----------------------------- fitting ----------------------------- #
    def _fit_calibrator(
        self,
        Y_cal_norm: NDArray[np.float64],  # decisions y (n, d_y)
        X_cal_norm: NDArray[np.float64],  # targets  x (n, d_x)
    ) -> None:
        """Compute the split-conformal L2 radius after estimator fitting."""
        Y_cal = np.asarray(Y_cal_norm, dtype=float)
        X_cal = np.asarray(X_cal_norm, dtype=float)

        if Y_cal.ndim != 2 or X_cal.ndim != 2:
            raise ValueError("Y_cal_norm and X_cal_norm must be 2-D (n, d)")
        if Y_cal.shape[0] != X_cal.shape[0]:
            raise ValueError("Y_cal_norm and X_cal_norm must have the same n")
        self._dy, self._dx = Y_cal.shape[1], X_cal.shape[1]

        X_hat = self.estimator.predict(Y_cal)  # (n, d_x)
        if X_hat.shape != X_cal.shape:
            raise ValueError(
                f"Estimator.predict returned shape {X_hat.shape}, expected {X_cal.shape}"
            )
        resid = np.linalg.norm(X_cal - X_hat, axis=1)  # (n,)

        self._n_cal = int(resid.shape[0])

        # finite-sample split-CP quantile: tau = ceil((n+1)*conf)/n (in [0,1])
        tau = float(np.ceil((self._n_cal + 1) * self._confidence) / self._n_cal)
        tau = min(max(tau, 0.0), 1.0)

        # use "higher" so coverage is at least the requested level
        self._radius = float(np.quantile(resid, tau, method="higher"))

    # ---------------------------- inference ---------------------------- #
    def transform(
        self,
        y_norm: NDArray[np.float64],
        x_target_norm: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64] | float]:
        """
        Compute prediction and conformal-aware distance summary.

        Args:
          y_norm        : (d_y,) or (n, d_y)   proposed decision(s)
          x_target_norm : (d_x,) or (n, d_x)   target objective(s) to hit

        Returns:
          {
            "x_hat"    : (n, d_x) predicted objectives,
            "d_l2"     : float, L2 distance summary (max over batch),
            "d_l2_all" : (n,) per-sample distances (for diagnostics),
          }
        """
        if self._radius is None:
            raise RuntimeError("Calibrator must be fitted before transform().")

        y = np.asarray(y_norm, dtype=float)
        x_t = np.asarray(x_target_norm, dtype=float)

        # ensure 2D
        if y.ndim == 1:
            y = y[None, :]
        if x_t.ndim == 1:
            x_t = x_t[None, :]

        # basic shape checks
        if self._dy is not None and y.shape[1] != self._dy:
            raise ValueError(f"y_norm has d_y={y.shape[1]}, expected {self._dy}")
        if self._dx is not None and x_t.shape[1] != self._dx:
            raise ValueError(
                f"x_target_norm has d_x={x_t.shape[1]}, expected {self._dx}"
            )
        if y.shape[0] != x_t.shape[0]:
            raise ValueError("Batch sizes differ between y_norm and x_target_norm")

        x_hat = self.estimator.predict(y)  # (n, d_x)
        if x_hat.shape != x_t.shape:
            raise ValueError(
                f"Estimator.predict returned shape {x_hat.shape}, expected {x_t.shape}"
            )

        d_all = np.linalg.norm(x_hat - x_t, axis=1)  # (n,)
        d_sum = float(d_all.max())  # conservative summary for batches

        return {
            "x_hat": x_hat.astype(np.float64, copy=False),
            "d_l2": d_sum,
            "d_l2_all": d_all.astype(np.float64, copy=False),
        }

    def evaluate(
        self,
        *,
        candidate: NDArray[np.float64],
        target: NDArray[np.float64],
        eps_l2: float | None,
        eps_per_obj: NDArray[np.float64] | Sequence[float] | None,
    ) -> tuple[bool, dict[str, float | bool], str]:
        """Evaluate the tolerance gate for a candidate decision."""
        transformed = self.transform(candidate, target)
        x_hat = transformed["x_hat"]
        x_hat_vec = self._as_vector(x_hat)
        target_vec = self._as_vector(np.asarray(target, dtype=float))

        diff = np.abs(x_hat_vec - target_vec)
        dist_l2 = float(transformed["d_l2"])
        q = self.radius

        covered = self._check_coverage(diff, dist_l2, q, eps_l2, eps_per_obj)

        metrics: dict[str, float | bool] = {
            "gate2_conformal_radius_q": q,
            "gate2_dist_to_target_l2": dist_l2,
            "gate2_covered": covered,
        }
        explanation = (
            "Pass: predicted X stays within tolerance after accounting for model error."
            if covered
            else "ABSTAIN: predicted X exceeds tolerance after accounting for model error."
        )
        return bool(covered), metrics, explanation

    # ----------------------------- accessor ---------------------------- #
    @property
    def radius(self) -> float:
        if self._radius is None:
            raise RuntimeError("SplitConformalL2Calibrator has not been fitted yet.")
        return self._radius

    @staticmethod
    def _as_vector(array: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(array, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim != 1:
            raise ValueError("Expected a single-sample vector for evaluation.")
        return arr

    @staticmethod
    def _check_coverage(
        diff: NDArray[np.float64],
        dist_l2: float,
        radius_q: float,
        eps_l2: float | None,
        eps_per_obj: NDArray[np.float64] | Sequence[float] | None,
    ) -> bool:
        if eps_l2 is None and eps_per_obj is None:
            raise ValueError("Provide eps_l2 or eps_per_obj to evaluate the gate.")

        if eps_l2 is not None:
            if eps_l2 < 0:
                raise ValueError("eps_l2 must be non-negative.")
            return bool((dist_l2 + radius_q) <= float(eps_l2))

        eps_array = np.asarray(eps_per_obj, dtype=float)
        if np.any(eps_array < 0):
            raise ValueError("eps_per_obj entries must be non-negative.")
        if eps_array.shape != diff.shape:
            raise ValueError("eps_per_obj shape does not match the target dimension.")
        return bool(np.all(diff + radius_q <= eps_array))
