"""Value objects describing calibration results for decision validation."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class OODCalibration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mu: NDArray[np.float64]
    prec: NDArray[np.float64]
    threshold_md2: float

    @field_validator("mu", "prec", mode="before")
    def _to_array(cls, value):  # type: ignore[override]
        return np.asarray(value, dtype=float)


class ConformalCalibration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    radius_q: float
