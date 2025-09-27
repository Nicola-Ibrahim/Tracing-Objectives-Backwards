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
      - fit(X, y)
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
    def fit(
        self,
        X: NDArray[np.float64],  # decisions (n, d_x)
        y: NDArray[np.float64],  # targets   (n, d_y)
    ) -> None:
        """Fit the estimator and compute the split-conformal L2 radius."""
        decisions = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)

        if decisions.ndim != 2 or targets.ndim != 2:
            raise ValueError("X and y must be 2-D (n, d)")
        if decisions.shape[0] != targets.shape[0]:
            raise ValueError("X and y must have the same n")

        self.estimator.fit(decisions, targets)

        self._dy, self._dx = decisions.shape[1], targets.shape[1]

        predicted_targets = self.estimator.predict(decisions)  # (n, d_y)
        if predicted_targets.shape != targets.shape:
            raise ValueError(
                f"Estimator.predict returned shape {predicted_targets.shape}, expected {targets.shape}"
            )
        resid = np.linalg.norm(targets - predicted_targets, axis=1)  # (n,)

        self._n_cal = int(resid.shape[0])

        # finite-sample split-CP quantile: tau = ceil((n+1)*conf)/n (in [0,1])
        tau = float(np.ceil((self._n_cal + 1) * self._confidence) / self._n_cal)
        tau = min(max(tau, 0.0), 1.0)

        # use "higher" so coverage is at least the requested level
        self._radius = float(np.quantile(resid, tau, method="higher"))

    # ---------------------------- inference ---------------------------- #
    def _prepare_evaluation(
        self,
        *,
        candidate: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64] | float]:
        """Return prediction diagnostics used by the evaluation step."""
        if self._radius is None:
            raise RuntimeError("Calibrator must be fitted before evaluation.")

        y = np.asarray(candidate, dtype=float)
        x_t = np.asarray(target, dtype=float)

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
        eps_l2: float,
        eps_per_obj: NDArray[np.float64] | Sequence[float],
    ) -> tuple[bool, dict[str, float | bool], str]:
        """Evaluate the tolerance gate for a candidate decision."""
        transformed = self._prepare_evaluation(candidate=candidate, target=target)
        x_hat = transformed["x_hat"]
        x_hat_vec = self._as_vector(x_hat)
        target_vec = self._as_vector(np.asarray(target, dtype=float))

        diff = np.abs(x_hat_vec - target_vec)
        dist_l2 = float(transformed["d_l2"])
        q = self.radius

        covered = self._check_coverage(diff, dist_l2, q, eps_l2, eps_per_obj)

        metrics: dict[str, float | bool] = {
            "conformal_radius_q": q,
            "dist_to_target_l2": dist_l2,
            "covered": covered,
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
