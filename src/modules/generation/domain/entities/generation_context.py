from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ....modeling.domain.interfaces.base_transform import BaseTransformer


class GenerationContext(BaseModel):
    """
    Represents the pre-computed geometric and statistical environment
    for real-time candidate generation against a specific dataset.
    Now structured with explicit normalizer transformations and embedded surrogate to map
    backward steps coherently.
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

    # Standardized transformers following the trained_pipeline pattern
    transforms: list[BaseTransformer] = Field(
        default_factory=list,
        description="Ordered sequence of fitted preprocessing transforms",
    )

    # Local Forward Surrogate
    surrogate_step: Any = Field(
        ..., description="The Explicitly Evaluated Surrogate Model Step"
    )

    is_trained: bool = Field(
        default=True,
        description="Flag indicating if the context has been fully trained",
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context):
        if self.objectives.ndim != 2 or self.objectives.shape[1] != 2:
            raise ValueError("objectives must be a 2D array with shape (N, 2)")
        if self.objectives.shape[0] < 4:
            raise ValueError(
                "objectives shape[0] must be >= 4 for non-degenerate 2D triangulation"
            )
        if self.anchors_norm.shape[0] != self.objectives.shape[0]:
            raise ValueError("anchors_norm must be row-aligned with objectives")

    def evaluate_simplex_reliability(self, anchor_candidates: np.ndarray) -> bool:
        """
        Domain service acting as a safeguard to prevent generation of physically impossible configurations.
        Evaluates whether a set of geometric vertices forming a facet is smaller than the local coherence threshold.

        Args:
            anchor_candidates: (N, D) array of normalized decision configurations for the anchors.

        Returns:
            True if coherent (all pairwise distances <= tau), False otherwise.
        """
        if len(anchor_candidates) < 2:
            return True

        # Calculate pairwise euclidean distances
        diffs = anchor_candidates[:, None, :] - anchor_candidates[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Get upper triangle, excluding diagonal
        i, j = np.triu_indices(len(anchor_candidates), k=1)
        pairwise_dists = dists[i, j]

        return bool(np.all(pairwise_dists <= self.tau))

    def get_decisions_transforms(self) -> list[BaseTransformer]:
        from ....modeling.domain.interfaces.base_transform import TransformTarget

        return [
            t
            for t in self.transforms
            if getattr(t, "target", None) in (TransformTarget.DECISIONS, "decisions")
        ]

    def get_objectives_transforms(self) -> list[BaseTransformer]:
        from ....modeling.domain.interfaces.base_transform import TransformTarget

        return [
            t
            for t in self.transforms
            if getattr(t, "target", None) in (TransformTarget.OBJECTIVES, "objectives")
        ]
