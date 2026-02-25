from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GenerationResult:
    """
    The ranked output of the generation pipeline.
    """

    candidates: np.ndarray  # (M, D) Array of decision-space candidates
    predicted_objectives: np.ndarray  # (M, 2) Array of predicted objectives
    residual_errors: np.ndarray  # (M,) Array of residual errors vs target
    pathway: Literal["coherent", "incoherent"]
    target_objective: np.ndarray  # (1, 2) The user-requested target
    anchor_indices: list[int]  # Indices of the Delaunay triangle anchors used
    is_inside_mesh: bool  # Whether the target fell inside the Delaunay mesh

    def __post_init__(self):
        M = self.candidates.shape[0]
        if self.predicted_objectives.shape[0] != M:
            raise ValueError(
                "candidates and predicted_objectives must have same number of rows"
            )
        if self.residual_errors.shape[0] != M:
            raise ValueError(
                "candidates and residual_errors must have same number of rows"
            )

        # Verify residual_errors are sorted ascending
        if not np.all(self.residual_errors[:-1] <= self.residual_errors[1:]):
            raise ValueError(
                "GenerationResult candidates must be sorted by residual error ascending"
            )
