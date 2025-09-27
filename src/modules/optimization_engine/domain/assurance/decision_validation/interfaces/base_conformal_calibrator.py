import inspect
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ....modeling.interfaces.base_estimator import BaseEstimator


class BaseConformalCalibrator(ABC):
    """Base contract for conformal calibrators coupled with an estimator."""

    def __init__(self, estimator: BaseEstimator) -> None:
        if estimator is None:
            raise ValueError("BaseConformalCalibrator requires an estimator instance.")
        self._estimator = estimator

    @property
    def estimator(self) -> BaseEstimator:
        """Return the estimator trained alongside the calibrator."""
        return self._estimator

    @property
    def estimator_type(self) -> str:
        """Return the type identifier for the attached estimator."""
        return self._estimator.type

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the calibrator."""

        return {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "init_params": self._collect_init_params(),
        }

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit the attached estimator and any calibrator-specific components."""

    @abstractmethod
    def _prepare_evaluation(
        self,
        *,
        candidate: np.ndarray,
        target: np.ndarray,
    ) -> Any: ...

    @abstractmethod
    def evaluate(
        self,
        *,
        candidate: np.ndarray,
        target: np.ndarray,
        eps_l2: float | None,
        eps_per_obj: Sequence[float] | np.ndarray | None,
    ) -> tuple[bool, dict[str, float | bool], str]: ...

    @property
    @abstractmethod
    def radius(self) -> float: ...

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

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
