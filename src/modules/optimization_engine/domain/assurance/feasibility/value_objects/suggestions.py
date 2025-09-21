"""Container for feasible objective suggestions."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from ...shared.ndarray_utils import ensure_2d


class Suggestions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: NDArray[np.float64]

    @field_validator("values", mode="before")
    def _coerce_array(cls, value):  # type: ignore[override]
        arr = ensure_2d(np.asarray(value, dtype=float))
        return arr

    @property
    def count(self) -> int:
        return int(self.values.shape[0])


__all__ = ["Suggestions"]
