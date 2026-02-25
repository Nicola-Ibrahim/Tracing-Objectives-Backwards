from typing import Optional

import numpy as np

from ..value_objects.generation_result import GenerationResult


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
