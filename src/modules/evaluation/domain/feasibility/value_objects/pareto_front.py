"""Lightweight wrapper around a Pareto front for invariant checks."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...shared.ndarray_utils import ensure_2d


class ParetoFront(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw: NDArray[np.float64]
    normalized: NDArray[np.float64]

    @field_validator("raw", "normalized", mode="before")
    def _coerce_array(cls, value):  # type: ignore[override]
        arr = np.asarray(value, dtype=float)
        return ensure_2d(arr)

    @model_validator(mode="after")
    def _shapes_match(self):
        if self.raw.shape != self.normalized.shape:
            raise ValueError(
                "Raw and normalised Pareto fronts must share the same shape."
            )
        return self

    @property
    def objective_dim(self) -> int:
        return int(self.raw.shape[1])

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.raw.min(axis=0), self.raw.max(axis=0)


__all__ = ["ParetoFront"]
