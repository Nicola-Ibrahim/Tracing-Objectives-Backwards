from typing import Any

from pydantic import BaseModel

from ..value_objects.accuracy_summary import AccuracySummary


class AccuracyLens(BaseModel):
    discrepancy_scores: Any  # np.ndarray
    best_shot_residuals: Any  # np.ndarray
    systematic_bias: Any  # np.ndarray
    cloud_dispersion: Any  # np.ndarray
    summary: AccuracySummary

    class Config:
        arbitrary_types_allowed = True
