from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

from ..interpolation.exceptions import (
    FeasibilityFailureReason,
    ObjectiveOutOfBoundsError,
)


class FeasibilityScoringStrategy(ABC):
    """Abstract base class for all feasibility scoring strategies."""

    @abstractmethod
    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """
        Computes a feasibility score for the target.
        Higher score means more feasible.
        Score should be in range [0,1].
        """
        raise NotImplementedError()

    def is_feasible(self, score: float, tolerance: float) -> bool:
        """Determines if a metric (score) is within the acceptable tolerance."""
        return score >= tolerance


class LocalSphereScoreStrategy(FeasibilityScoringStrategy):
    """
    Strategy based on Euclidean distance, forming a sphere around each Pareto point.
    A target's feasibility is scored based on its distance from the closest point.
    """

    def __init__(self, radius: float = 1.0):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._radius = radius

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """
        Computes a score from 0 to 1. A score of 1 means the target is identical
        to a Pareto point. A score of 0 means it's on or beyond the boundary of the sphere.
        """
        distances = np.linalg.norm(pareto_points - target, axis=1)
        scores = np.clip(1 - distances / self._radius, 0.0, 1.0)
        return float(np.max(scores))


class KDEScoreStrategy(FeasibilityScoringStrategy):
    """
    Strategy using Kernel Density Estimation (KDE) to create a continuous
    feasibility field from the Pareto points.
    """

    def __init__(self, bandwidth: float = 0.1):
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        self._bandwidth = bandwidth
        self._max_pareto_density = None

    def _get_density(self, points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """Helper to calculate Gaussian kernel density."""
        diff = points[:, np.newaxis, :] - ref_points[np.newaxis, :, :]
        sq_distances = np.sum(diff**2, axis=-1)
        kernel_vals = np.exp(-sq_distances / (2 * self._bandwidth**2))
        return np.sum(kernel_vals, axis=1)

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """
        Computes the KDE value at the target, normalized by the maximum KDE
        value on the Pareto front itself.
        """
        target_density = self._get_density(target, pareto_points)

        # Lazily compute and cache max density on the front
        if self._max_pareto_density is None:
            pareto_densities = self._get_density(pareto_points, pareto_points)
            self._max_pareto_density = np.max(pareto_densities)

        if self._max_pareto_density == 0:
            return 0.0

        return float(target_density[0] / self._max_pareto_density)


class ConvexHullScoreStrategy(FeasibilityScoringStrategy):
    """
    Strategy based on the distance to the convex hull of the Pareto front.
    Best for low-dimensional spaces due to computational complexity.
    """

    def __init__(self, delta: float):
        if delta <= 0:
            raise ValueError("Delta must be positive.")
        self._delta = delta

    def _distance_to_hull(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """Calculates Euclidean distance from a target to the convex hull of points."""
        num_points = pareto_points.shape[0]
        objective = lambda w: np.linalg.norm(w @ pareto_points - target)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, None) for _ in range(num_points))
        w_init = np.ones(num_points) / num_points
        result = minimize(
            objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints
        )
        return float(result.fun)

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        distance = self._distance_to_hull(target.flatten(), pareto_points)
        score = max(0.0, 1 - distance / self._delta)
        return score


class MinDistanceScoreStrategy(FeasibilityScoringStrategy):
    """
    Simplest strategy: converts minimum Euclidean distance to a feasibility score.
    """

    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("Delta must be positive.")
        self._delta = delta

    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        distances = np.linalg.norm(pareto_points - target, axis=1)
        min_distance = np.min(distances)
        score = max(0.0, 1 - min_distance / self._delta)
        return score


class ObjectiveFeasibilityChecker:
    def __init__(
        self,
        pareto_front: np.ndarray,
        pareto_front_norm: np.ndarray,
        tolerance: float,
        scorer: FeasibilityScoringStrategy,
    ):
        """
        Initializes the ObjectiveFeasibilityChecker.

        Args:
            pareto_front (np.ndarray): Unnormalized historical Pareto front points with shape (num_points, num_objectives).
            pareto_front_norm (np.ndarray): Corresponding normalized Pareto front points scaled to [0,1] with same shape.
            tolerance (float): Threshold for feasibility score; target is considered feasible if score >= tolerance.
            scorer (FeasibilityScoringStrategy): Instance of a feasibility scoring strategy used to compute feasibility metric.

        Raises:
            ValueError: If the normalized and unnormalized Pareto fronts do not have matching feature dimensions.
        """
        if pareto_front.shape[1] != pareto_front_norm.shape[1]:
            raise ValueError(
                "Normalized and unnormalized fronts must have the same feature dimension."
            )

        self._pareto_front = pareto_front
        self._pareto_front_norm = pareto_front_norm
        self._tolerance = tolerance
        self._scorer = scorer
        self._min_unnorm_bounds = pareto_front.min(axis=0)
        self._max_unnorm_bounds = pareto_front.max(axis=0)

    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        """Ensure input is 2D for consistent vectorized operations."""
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def _is_within_objective_bounds(self, target: np.ndarray) -> bool:
        """
        Returns True if target is within all objective bounds; False otherwise.
        """
        target = self._ensure_2d(target)
        for i in range(target.shape[1]):
            val = target[0, i]
            min_val, max_val = self._min_unnorm_bounds[i], self._max_unnorm_bounds[i]
            if not (min_val <= val <= max_val):
                return False
        return True

    def _compute_feasibility_score(self, target_norm: np.ndarray) -> float:
        """Compute the feasibility score for the normalized target."""
        target_norm = self._ensure_2d(target_norm)
        return self._scorer.compute_metric(target_norm, self._pareto_front_norm)

    def _is_score_feasible(self, score: float) -> bool:
        """Check if the computed score passes the tolerance threshold."""
        return self._scorer.is_feasible(score, self._tolerance)

    def _raise_out_of_bounds_error(
        self,
        target_norm: np.ndarray,
        message: str,
        reason: FeasibilityFailureReason,
        score: float | None = None,
        extra_info: str | None = None,
    ) -> None:
        """Helper to raise ObjectiveOutOfBoundsError with suggestions."""
        raise ObjectiveOutOfBoundsError(
            message=message,
            reason=reason,
            score=score,
            suggestions=self.get_nearest_suggestions(target_norm, 3),
            extra_info=extra_info,
        )

    def get_nearest_suggestions(self, target_norm: np.ndarray, num: int) -> np.ndarray:
        """
        Find the 'num' closest points on the normalized Pareto front to the target_norm,
        with small random perturbations to encourage exploration.
        """
        if num <= 0:
            return np.array([])
        target_norm = self._ensure_2d(target_norm)
        distances = np.linalg.norm(self._pareto_front_norm - target_norm, axis=1)
        nearest_indices = np.argsort(distances)[:num]
        suggestions = self._pareto_front_norm[nearest_indices]
        perturbation_scale = self._tolerance / 5.0
        perturbations = np.random.uniform(
            -perturbation_scale, perturbation_scale, size=suggestions.shape
        )
        return np.clip(suggestions + perturbations, 0.0, 1.0)

    def validate(
        self, target: np.ndarray, target_norm: np.ndarray, num_suggestions: int = 3
    ) -> None:
        """
        Validate if a target objective is feasible with respect to the historical Pareto front.

        Raises:
            ObjectiveOutOfBoundsError with suggestions if validation fails.
        """
        target = self._ensure_2d(target)
        target_norm = self._ensure_2d(target_norm)

        if not self._is_within_objective_bounds(target):
            violations = []
            for i in range(target.shape[1]):
                val = target[0, i]
                min_val, max_val = (
                    self._min_unnorm_bounds[i],
                    self._max_unnorm_bounds[i],
                )
                if not (min_val <= val <= max_val):
                    violations.append(
                        f"Feature {i}: value = {val:.4f}, allowed range = [{min_val:.4f}, {max_val:.4f}]"
                    )
            extra_info = "Target objective out of bounds:\n" + "\n".join(violations)
            self._raise_out_of_bounds_error(
                target_norm,
                message="Target objective is outside the historical Pareto front bounds.",
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                score=float("inf"),
                extra_info=extra_info,
            )

        score = self._compute_feasibility_score(target_norm)

        if not self._is_score_feasible(score):
            self._raise_out_of_bounds_error(
                target_norm,
                message="Target objective is not feasible according to the chosen strategy.",
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                score=score,
                extra_info=f"Computed score = {score:.4f}, required tolerance = {self._tolerance:.4f}.",
            )
