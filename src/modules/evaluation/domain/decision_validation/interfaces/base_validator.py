import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


   


class BaseConformalValidator(ABC):
    """
    Base contract for conformal validators operating on already predicted objectives.

    Required subclass API:
      - fit(y_pred_cal, y_true_cal) -> None
      - validate(y_pred, y_target, tolerance) -> (passed, metrics, explanation)
    """

    def __init__(self, confidence: float = 0.90) -> None:
        """
        Args:
            confidence: desired coverage level in (0,1); e.g., 0.90 ≈ 90th pct.

        Raises:
            ValueError: if confidence not in (0,1).
        """
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must lie in (0,1)")
        self._confidence = float(confidence)
        self._n_cal: int = 0
        self._y_dim: int | None = None  # dim of objective

        self._radius_q: float = None  # learned cushion (quantile radius)

    @property
    def radius(self) -> float:
        """Return the learned conformal radius (safety cushion)."""
        if not hasattr(self, "_radius_q") or self._radius_q is None:
            raise ValueError("Validator has not been fitted yet; radius is undefined.")
        return self._radius_q

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the validator."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "init_params": self._collect_init_params(),
        }

    def describe(self) -> dict[str, Any]:
        return self.to_json()


    @abstractmethod
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit the conformal validator from calibration predictions and ground-truth.
        Shapes:
          - y_pred: (n, d_y)
          - y_true: (n, d_y)
        """

    # ----------------------------- abstract API ----------------------------- #

    @abstractmethod
    def validate(
        self, y_pred: np.ndarray, y_target: np.ndarray, tolerance: float
    ) -> tuple[bool, dict[str, float | bool], str]:
        """Validate prediction(s) against the target objective."""

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
        if isinstance(value, (list, tuple)):
            return [self._simplify_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._simplify_value(v) for k, v in value.items()}
        if isinstance(value, _np.ndarray):
            return value.tolist()
        return str(value)


class BaseOODValidator(ABC):  
    """
    Base contract for OOD validators.

    Required subclass API:
      - fit(X) -> None
      - validate(X) -> (passed, metrics, explanation)
      - inlier_threshold_sq (float property)
    """

    @abstractmethod
    def validate(self, X: np.ndarray) -> tuple[bool, dict[str, float | bool], str]:
        """
        Decide inlier/outlier for X (or a batch) under the learned inlier criterion.
        """

    def evaluate(self, X: np.ndarray) -> tuple[bool, dict[str, float | bool], str]:
        """Backward-compatible alias for validate()."""
        return self.validate(X)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the validator."""
        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "init_params": self._collect_init_params(),
        }

    # --------------------------- introspection utils -------------------------- #
    @property
    def threshold(self) -> float:
        """Return the learned OOD threshold."""
        if not hasattr(self, "_threshold") or self._threshold is None:
            raise ValueError(
                "Validator has not been fitted yet; threshold is undefined."
            )
        return self._threshold

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
        if isinstance(value, (list, tuple)):
            return [self._simplify_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._simplify_value(v) for k, v in value.items()}
        if isinstance(value, _np.ndarray):
            return value.tolist()
        return str(value)
