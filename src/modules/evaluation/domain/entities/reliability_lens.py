from typing import Any

from pydantic import BaseModel

from ..value_objects.calibration_curve import CalibrationCurve
from ..value_objects.reliability_summary import ReliabilitySummary


class ReliabilityLens(BaseModel):
    pit_values: Any  # np.ndarray
    calibration_error: float
    crps: float
    diversity: Any  # np.ndarray
    interval_width: Any  # np.ndarray
    summary: ReliabilitySummary
    calibration_curve: CalibrationCurve

    class Config:
        arbitrary_types_allowed = True
