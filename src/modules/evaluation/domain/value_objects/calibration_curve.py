from typing import Any

from pydantic import BaseModel


class CalibrationCurve(BaseModel):
    pit_values: Any  # np.ndarray
    cdf_y: Any

    class Config:
        arbitrary_types_allowed = True
