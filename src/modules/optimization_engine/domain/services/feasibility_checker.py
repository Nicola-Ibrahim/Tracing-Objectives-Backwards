import numpy as np

from ..interpolation.exceptions import (
    FeasibilityFailureReason,
    ObjectiveOutOfBoundsError,
)


class ObjectiveFeasibilityChecker:
    """
    A soft validator for checking if a target objective (in objective space)
    lies near the known Pareto front.

    It avoids hard geometric constraints (like convex hulls) and instead:
    - Ensures the target is within the raw bounds of past Pareto-optimal solutions.
    - Uses Euclidean distance in normalized space to assess closeness to feasibility.

    If the distance exceeds a specified tolerance, the checker raises
    an ObjectiveOutOfBoundsError with suggestions for nearby feasible points.
    """

    def __init__(
        self,
        pareto_front: np.ndarray,  # Raw (unnormalized) Pareto front
        pareto_front_norm: np.ndarray,  # Normalized Pareto front
        tolerance: float,  # Allowed Euclidean distance in normalized space
    ):
        if pareto_front.shape[1] != pareto_front_norm.shape[1]:
            raise ValueError(
                "Normalized and unnormalized fronts must have same feature dimension."
            )

        self._pareto_front = pareto_front
        self._pareto_front_norm = pareto_front_norm
        self._tolerance = tolerance

        self._min_unnorm_bounds = pareto_front.min(axis=0)
        self._max_unnorm_bounds = pareto_front.max(axis=0)

    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        """Ensures input array is 2D (n_samples, n_features)."""
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def _euclidean_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance between rows of A and a single row in B."""
        return np.linalg.norm(A - B, axis=1)

    def _check_within_unnormalized_objective_bounds(self, target: np.ndarray) -> None:
        """
        Checks if target objective lies within the min/max of the unnormalized Pareto front.
        Raises ValueError with a detailed message if any feature is out of bounds.
        """
        target = self._reshape(target)

        violations = []
        for i in range(target.shape[1]):
            val = target[0, i]
            min_val = self._min_unnorm_bounds[i]
            max_val = self._max_unnorm_bounds[i]
            if val < min_val or val > max_val:
                violations.append(
                    f"Feature {i}: value = {val:.4f}, allowed range = [{min_val:.4f}, {max_val:.4f}]"
                )

        if violations:
            raise ValueError(
                "Target objective out of bounds:\n" + "\n".join(violations)
            )

    def _get_distance_to_nearest_pareto_point(self, target_norm: np.ndarray) -> float:
        """
        Computes the minimum Euclidean distance between the normalized target objective
        and any point in the normalized Pareto front.
        """
        target_norm = self._reshape(target_norm)
        return float(
            np.min(self._euclidean_distance(self._pareto_front_norm, target_norm))
        )

    def get_nearest_suggestions(self, target_norm: np.ndarray, num: int) -> np.ndarray:
        """
        Suggests the `num` closest feasible points (normalized) to the target.
        Adds slight random perturbations and clips to [0,1] to ensure diversity and validity.
        """
        if num <= 0:
            return np.array([])

        target_norm = self._reshape(target_norm)
        distances = self._euclidean_distance(self._pareto_front_norm, target_norm)
        nearest_indices = np.argsort(distances)[:num]

        suggestions = self._pareto_front_norm[nearest_indices]

        # Perturb the suggestions to avoid duplicates and increase robustness
        perturbation_scale = self._tolerance / 5.0  # Example: 20% of tolerance
        perturbations = np.random.uniform(
            low=-perturbation_scale,
            high=perturbation_scale,
            size=suggestions.shape,
        )

        perturbed = suggestions + perturbations
        perturbed = np.clip(perturbed, 0.0, 1.0)

        return perturbed

    def validate(
        self,
        target: np.ndarray,  # Unnormalized objective (used for bounds)
        target_norm: np.ndarray,  # Normalized objective (used for distance)
        num_suggestions: int = 3,
    ) -> None:
        """
        Validates the feasibility of a target objective.

        Steps:
        1. Rejects immediately if target is outside unnormalized bounds.
        2. Accepts if the normalized target is within a specified Euclidean distance
           from any known Pareto-optimal point (i.e., tolerance).
        3. Raises an ObjectiveOutOfBoundsError if outside tolerance, suggesting alternatives.
        """
        target = self._reshape(target)
        target_norm = self._reshape(target_norm)

        # Step 1: Raw bounds check
        try:
            self._check_within_unnormalized_objective_bounds(target)
        except ValueError as e:
            raise ObjectiveOutOfBoundsError(
                message="Target objective is outside the historical Pareto front bounds.",
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                distance=float("inf"),
                suggestions=self.get_nearest_suggestions(target_norm, num_suggestions),
                extra_info=str(e),
            )

        # Step 2: Soft proximity check in normalized space
        distance = self._get_distance_to_nearest_pareto_point(target_norm)
        if distance > self._tolerance:
            raise ObjectiveOutOfBoundsError(
                message="Target objective is not close enough to the Pareto front.",
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                distance=distance,
                suggestions=self.get_nearest_suggestions(target_norm, num_suggestions),
                extra_info=(
                    f"Distance to closest Pareto point: {distance:.4f}, "
                    f"but allowed tolerance is {self._tolerance:.4f}. "
                    "Try adjusting your objective slightly."
                ),
            )

        # ✅ Valid — the target is close enough to the Pareto front
