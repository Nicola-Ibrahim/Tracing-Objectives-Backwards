from typing import Any, Optional

from pydantic import BaseModel

from ..value_objects.accuracy_summary import AccuracySummary


class AccuracyLens(BaseModel):
    discrepancy_scores: Any  # np.ndarray
    best_shot_scores: Optional[Any] = None  # np.ndarray
    rank_indices: Optional[Any] = None  # np.ndarray
    systematic_bias: Any  # np.ndarray
    cloud_dispersion: Any  # np.ndarray
    summary: AccuracySummary

    class Config:
        arbitrary_types_allowed = True
