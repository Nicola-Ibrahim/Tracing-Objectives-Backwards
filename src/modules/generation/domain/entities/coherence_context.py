from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CoherenceContext(BaseModel):
    """
    Represents the pre-computed geometric and statistical environment
    for real-time candidate generation against a specific dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dataset_name: str = Field(..., description="Identifier of the source dataset")
    objectives: np.ndarray = Field(
        ..., description="Objective-space points (N x 2), used as Delaunay vertices"
    )
    anchors_norm: np.ndarray = Field(
        ..., description="Normalized decision-space points (N x D)"
    )
    tau: float = Field(..., gt=0, description="Coherence threshold")
    k_neighbors: int = Field(
        default=5, ge=1, description="Number of neighbors used for tau computation"
    )
    surrogate_type: str = Field(
        ..., description="Estimator type of the forward surrogate"
    )
    surrogate_version: int = Field(
        ..., description="Version of the persisted forward surrogate"
    )
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context):
        # Validation Rules from data-model.md
        if self.objectives.ndim != 2 or self.objectives.shape[1] != 2:
            raise ValueError("objectives must be a 2D array with shape (N, 2)")
        if self.objectives.shape[0] < 4:
            raise ValueError(
                "objectives shape[0] must be >= 4 for non-degenerate 2D triangulation"
            )
        if self.anchors_norm.shape[0] != self.objectives.shape[0]:
            raise ValueError("anchors_norm must be row-aligned with objectives")
