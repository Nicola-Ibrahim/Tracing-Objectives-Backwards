# ----------------------------- BaseOODCalibrator -----------------------------

import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseOODCalibrator(ABC):
    """
    Base contract for OOD calibrators.

    Required subclass API:
      - fit(X) -> None
      - evaluate(X) -> (passed, metrics, explanation)
      - inlier_threshold_sq (float property)
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the inlier region from calibration data X, shape (n, d).
        """

    @abstractmethod
    def evaluate(self, X: np.ndarray) -> tuple[bool, dict[str, float | bool], str]:
        """
        Decide inlier/outlier for X (or a batch) under the learned inlier criterion.
        """

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the calibrator."""
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
                "Calibrator has not been fitted yet; threshold is undefined."
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
