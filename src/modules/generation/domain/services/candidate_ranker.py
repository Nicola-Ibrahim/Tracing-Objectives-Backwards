from dataclasses import dataclass
from typing import Literal, Optional

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


class CandidateRanker:
    """
    Domain service for comparing predicted objectives against a target and ranking candidates.
    """

    @staticmethod
    def rank(
        candidates: np.ndarray,
        predicted_objectives: np.ndarray,
        target_objective: np.ndarray,
        pathway: str,
        anchor_indices: list[int],
        is_inside_mesh: bool,
        error_threshold: Optional[float] = None,
    ) -> GenerationResult:
        """
        Ranks candidates by residual error and applies optional quality filtering.

        Args:
            candidates: (M, D) array of generated candidates.
            predicted_objectives: (M, 2) array of surrogate-predicted objectives.
            target_objective: (1, 2) or (2,) array of the requested target.
            pathway: "coherent" or "incoherent"
            anchor_indices: List of anchor indices used to support generation
            is_inside_mesh: True if target objective was inside the known mesh
            error_threshold: Optional cutoff for maximum residual error

        Returns:
            GenerationResult containing ranked candidates.
        """
        target = np.asarray(target_objective).reshape(1, 2)

        # Calculate Euclidean residual errors between prediction and target
        errors = np.linalg.norm(predicted_objectives - target, axis=1)

        # Filter if threshold provided
        if error_threshold is not None:
            valid_mask = errors <= error_threshold
            candidates = candidates[valid_mask]
            predicted_objectives = predicted_objectives[valid_mask]
            errors = errors[valid_mask]

        # Sort descending by quality (ascending by error)
        sort_indices = np.argsort(errors)

        return GenerationResult(
            candidates=candidates[sort_indices],
            predicted_objectives=predicted_objectives[sort_indices],
            residual_errors=errors[sort_indices],
            pathway=pathway,
            target_objective=target,
            anchor_indices=anchor_indices,
            is_inside_mesh=is_inside_mesh,
        )
