from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class RankingResult:
    """
    The ranked output of the generation pipeline.
    """

    candidates: np.ndarray  # (M, D) Final decision-space candidates (Raw/Un-normalized)
    predicted_objectives: np.ndarray  # (M, 2) Predicted objectives (Raw/Un-normalized)
    residual_errors: (
        np.ndarray
    )  # (M,) Array of residual errors vs target (Calculated in Normalized space)
    pathway: Literal["coherent", "incoherent"]
    anchor_indices: list[int]  # Indices of the Delaunay triangle anchors used
    is_inside_mesh: bool  # Whether the target fell inside the Delaunay mesh
    winner_index: int  # Index of the best candidate in this result
    winner_point: np.ndarray  # (1, 2) The best performance vector (Raw/Un-normalized)
    winner_decision: np.ndarray  # (1, D) The best decision vector (Raw/Un-normalized)

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
                "RankingResult candidates must be sorted by residual error ascending"
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
        objective_transforms: list = None,
        decision_transforms: list = None,
        error_threshold: Optional[float] = None,
    ) -> RankingResult:
        """
        Ranks candidates by residual error, filters if needed, and un-normalizes results.

        Args:
            candidates: (M, D) array of generated candidates (Normalized space).
            predicted_objectives: (M, 2) array of surrogate-predicted objectives (Normalized space).
            target_objective: (1, 2) or (2,) array of the requested target (Normalized space).
            pathway: "coherent" or "incoherent"
            anchor_indices: List of anchor indices used to support generation
            is_inside_mesh: True if target objective was inside the known mesh
            objective_transforms: Ordered list of transformers for objectives.
            decision_transforms: Ordered list of transformers for decisions.
            error_threshold: Optional cutoff for maximum residual error (Normalized space).

        Returns:
            RankingResult containing ranked, UN-NORMALIZED candidates.
        """
        target = np.asarray(target_objective).reshape(1, 2)

        # 1. Calculate Euclidean residual errors (Normalized space)
        errors = np.linalg.norm(predicted_objectives - target, axis=1)

        # 2. Filter (Normalized space)
        if error_threshold is not None:
            valid_mask = errors <= error_threshold
            candidates = candidates[valid_mask]
            predicted_objectives = predicted_objectives[valid_mask]
            errors = errors[valid_mask]

        # 3. Sort (Normalized space)
        sort_indices = np.argsort(errors)
        candidates_sorted = candidates[sort_indices]
        objectives_sorted = predicted_objectives[sort_indices]
        errors_sorted = errors[sort_indices]

        # 4. Identify Winner (Normalized space)
        if len(sort_indices) > 0:
            winner_idx = 0  # Ranking is already sorted ascending by error
            best_objective_norm = objectives_sorted[winner_idx].reshape(1, 2)
            best_decision_norm = candidates_sorted[winner_idx].reshape(1, -1)
            original_winner_index = int(sort_indices[0])
        else:
            best_objective_norm = np.zeros((1, 2))
            best_decision_norm = np.zeros((1, candidates.shape[1]))
            original_winner_index = -1

        # 5. Inverse Transformation (Convert to RAW space)
        candidates_raw = candidates_sorted.copy()
        if decision_transforms:
            for t in reversed(decision_transforms):
                candidates_raw = t.inverse_transform(candidates_raw)

        objectives_raw = objectives_sorted.copy()
        if objective_transforms:
            for t in reversed(objective_transforms):
                objectives_raw = t.inverse_transform(objectives_raw)

        winner_point_raw = best_objective_norm.copy()
        if objective_transforms:
            for t in reversed(objective_transforms):
                winner_point_raw = t.inverse_transform(winner_point_raw)

        winner_decision_raw = best_decision_norm.copy()
        if decision_transforms:
            for t in reversed(decision_transforms):
                winner_decision_raw = t.inverse_transform(winner_decision_raw)

        # 6. Finalize RankingResult (Immutable)
        return RankingResult(
            candidates=candidates_raw,
            predicted_objectives=objectives_raw,
            residual_errors=errors_sorted,
            pathway=pathway,
            anchor_indices=anchor_indices,
            is_inside_mesh=is_inside_mesh,
            winner_index=original_winner_index,
            winner_point=winner_point_raw,
            winner_decision=winner_decision_raw,
        )
