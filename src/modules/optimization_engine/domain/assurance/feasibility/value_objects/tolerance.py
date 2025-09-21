"""Tolerance configuration for feasibility evaluation."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Tolerance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    eps_l2: float | None = None
    eps_per_obj: NDArray[np.float64] | None = None

    @field_validator("eps_l2")
    def _non_negative(cls, value: float | None):  # type: ignore[override]
        if value is not None and value < 0:
            raise ValueError("eps_l2 must be non-negative.")
        return value

    @field_validator("eps_per_obj", mode="before")
    def _coerce_array(cls, value):  # type: ignore[override]
        if value is None:
            return value
        arr = np.asarray(value, dtype=float)
        if np.any(arr < 0):
            raise ValueError("eps_per_obj entries must be non-negative.")
        return arr

    @model_validator(mode="after")
    def _ensure_any(self):
        if self.eps_l2 is None and self.eps_per_obj is None:
            raise ValueError("Provide at least one tolerance (eps_l2 or eps_per_obj).")
        return self


__all__ = ["Tolerance"]
