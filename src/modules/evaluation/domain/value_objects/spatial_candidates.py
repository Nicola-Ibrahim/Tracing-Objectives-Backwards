from typing import Any

from pydantic import BaseModel


class SpatialCandidates(BaseModel):
    ordered: Any  # np.ndarray
    median: Any  # np.ndarray
    std: Any  # np.ndarray

    class Config:
        arbitrary_types_allowed = True
