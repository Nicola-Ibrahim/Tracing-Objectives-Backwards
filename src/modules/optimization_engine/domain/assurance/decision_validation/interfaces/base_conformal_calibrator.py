import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ....modeling.interfaces.base_estimator import BaseEstimator


class BaseConformalCalibrator(ABC):
    """
    Base contract for conformal calibrators coupled with a forward estimator f_hat: y -> X.

    Required subclass API:
      - fit(y_cal, X_cal) -> None
      - evaluate(y=..., X_target=..., acceptance_threshold=...) -> (passed, metrics, explanation)
      - calibration_margin (float property)
    """

    def __init__(self, *, estimator: BaseEstimator, confidence: float = 0.90) -> None:
        """
        Args:
            estimator: forward mapper f_hat with .fit(y, X) and .predict(y) -> X.
            confidence: desired coverage level in (0,1); e.g., 0.90 ≈ 90th pct.

        Raises:
            ValueError: if confidence not in (0,1).
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0,1)")
        self._confidence = float(confidence)
        self._n_cal: int = 0
        self._x_dim: int | None = None  # dim of X
        self._y_dim: int | None = None  # dim of y

        self._estimator = estimator
        self._radius_q: float = None  # learned cushion (quantile radius)
        self._threshold: float = None  # learned OOD threshold

    @property
    def estimator(self) -> BaseEstimator:
        """Return the estimator trained alongside the calibrator."""
        return self._estimator

    @property
    def estimator_type(self) -> str:
        """Return the type identifier for the attached estimator."""
        return self._estimator.type

    @property
    def radius(self) -> float:
        """Return the learned conformal radius (safety cushion)."""
        if not hasattr(self, "_radius_q") or self._radius_q is None:
            raise ValueError("Calibrator has not been fitted yet; radius is undefined.")
        return self._radius_q

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the calibrator."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "init_params": self._collect_init_params(),
        }

    # ----------------------------- abstract API ----------------------------- #

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the attached estimator on (y, X) and learn any conformal statistics.
        Shapes:
          - y_cal: (n, d_y)
          - X_cal: (n, d_x)
        """

    @abstractmethod
    def evaluate(
        self,
        *,
        y: np.ndarray,
        X_target: np.ndarray,
        tolerance: float,
    ) -> tuple[bool, dict[str, float | bool], str]:
        """
        Evaluate a candidate y against X_target using a single global acceptance threshold.
        Returns:
          - passed: bool
          - metrics: dict of floats/bools
          - explanation: human-friendly string
        """

    # --------------------------- introspection utils -------------------------- #

    def _collect_init_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        signature = inspect.signature(self.__class__.__init__)
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            value = self._resolve_attribute(name, parameter)
            if value is not None:
                params[name] = self._simplify_value(value)
        return params

    def _resolve_attribute(self, name: str, parameter: inspect.Parameter) -> Any | None:
        if hasattr(self, name):
            return getattr(self, name)
        private_name = f"_{name}"
        if hasattr(self, private_name):
            return getattr(self, private_name)
        if parameter.default is not inspect._empty:
            return parameter.default
        return None

    def _simplify_value(self, value: Any) -> Any:
        import numpy as _np

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, BaseEstimator):
            return {
                "type": value.__class__.__name__,
                "module": value.__class__.__module__,
                "parameters": value.to_dict(),
            }
        if isinstance(value, (list, tuple)):
            return [self._simplify_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._simplify_value(v) for k, v in value.items()}
        if isinstance(value, _np.ndarray):
            return value.tolist()
        return str(value)
