"""Immutable representation of an objective vector in raw and normalised space."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...shared.ndarray_utils import ensure_2d


class ObjectiveVector(BaseModel):
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
                "Raw and normalized objective vectors must share the same shape."
            )
        return self

    @classmethod
    def from_raw(
        cls,
        raw: NDArray[np.float64],
        *,
        normalizer,
    ) -> "ObjectiveVector":
        raw_2d = ensure_2d(np.asarray(raw, dtype=float))
        normalized = normalizer.transform(raw_2d)
        return cls(raw=raw_2d, normalized=normalized)


__all__ = ["ObjectiveVector"]
