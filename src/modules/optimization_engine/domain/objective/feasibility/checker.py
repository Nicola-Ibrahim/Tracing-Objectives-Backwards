import numpy as np

from .enums import FeasibilityFailureReason
from .exceptions import ObjectiveOutOfBoundsError
from .scoring_strategies.base import FeasibilityScoringStrategy


class ObjectiveFeasibilityChecker:
    """
    Domain service responsible for validating the feasibility of a given target
    objective against a historical Pareto front.

    It performs checks based on raw objective bounds and a calculated feasibility score,
    raising an exception with diagnostic information and suggestions if the target
    is deemed infeasible.
    """

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
            pareto_front (np.ndarray): The historical Pareto front in its original,
                                       unnormalized objective space. Expected shape (N, D),
                                       where N is the number of points and D is the number of objectives.
            pareto_front_norm (np.ndarray): The historical Pareto front, normalized to a
                                            standard range (e.g., [0, 1] per objective).
                                            Expected shape (N, D).
            tolerance (float): A threshold value used by the scoring strategy to determine
                               if a score indicates feasibility. A score below this tolerance
                               suggests infeasibility.
            scorer (FeasibilityScoringStrategy): An instance of a scoring strategy
                                                 (e.g., KDE, MinDistance) to compute
                                                 the feasibility score.

        Raises:
            ValueError: If the unnormalized and normalized Pareto fronts have
                        different dimensionalities.
        """
        if pareto_front.shape[1] != pareto_front_norm.shape[1]:
            raise ValueError(
                "Pareto front and normalized front must have same dimensionality."
            )

        self._pareto_front = pareto_front
        self._pareto_front_norm = pareto_front_norm
        self._tolerance = tolerance
        self._scorer = scorer

        # Calculate and store the min/max bounds of the historical Pareto front
        # in the original objective space. Used for raw bound checking.
        self._bounds_min = pareto_front.min(axis=0)
        self._bounds_max = pareto_front.max(axis=0)

    # -----------------------
    # Utils
    # -----------------------

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """
        Ensures a NumPy array is 2-dimensional (e.g., [1, D] for a single point)
        for consistent array operations.

        Args:
            arr (np.ndarray): The input array, can be 1D (D,) or 2D (1, D).

        Returns:
            np.ndarray: The reshaped 2D array.
        """
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def _within_bounds(self, target: np.ndarray) -> list[str]:
        """
        Checks if the target objective point falls within the min/max bounds
        of the historical Pareto front (in original unnormalized space).

        Args:
            target (np.ndarray): The target objective point in its original scale.

        Returns:
            list[str]: A list of violation messages if any objective dimension
                       is out of bounds. Returns an empty list if all are within bounds.
        """
        target = self._ensure_2d(target)
        violations = []

        # Iterate through each objective dimension to check against historical bounds
        for i in range(target.shape[1]):
            val = target[0, i]
            if not (self._bounds_min[i] <= val <= self._bounds_max[i]):
                violations.append(
                    f"Feature {i}: {val:.4f} not in [{self._bounds_min[i]:.4f}, {self._bounds_max[i]:.4f}]"
                )
        return violations

    def _generate_suggestions(
        self, target_norm: np.ndarray, num: int = 3
    ) -> np.ndarray:
        """
        Generates `num` suggestions for feasible objective points based on the
        closest historical Pareto front points in the normalized space.
        A small random perturbation is added to encourage exploration around
        the known feasible points.

        Args:
            target_norm (np.ndarray): The infeasible target objective point in
                                      normalized space.
            num (int): The desired number of suggestions.

        Returns:
            np.ndarray: A 2D array of suggested feasible objective points in
                        normalized space, clipped to [0, 1].
        """
        target_norm = self._ensure_2d(target_norm)

        # Calculate Euclidean distances from the target to all normalized Pareto points
        distances = np.linalg.norm(self._pareto_front_norm - target_norm, axis=1)

        # Get indices of the 'num' closest Pareto points
        indices = np.argsort(distances)[:num]
        nearest_pareto_points = self._pareto_front_norm[indices]

        # Add slight random noise to the nearest points for diversity.
        # The noise range is relative to the internal tolerance.
        noise = np.random.uniform(
            -self._tolerance / 5, self._tolerance / 5, size=nearest_pareto_points.shape
        )

        # Apply noise and clip the results to ensure they remain within the normalized [0, 1] range.
        return np.clip(nearest_pareto_points + noise, 0.0, 1.0)

    def _raise_error(
        self,
        *,
        target_norm: np.ndarray,
        message: str,
        reason: FeasibilityFailureReason,
        score: float | None = None,
        extra_info: str | None = None,
        num_suggestions: int = 3,
    ) -> None:
        """
        Helper method to raise an ObjectiveOutOfBoundsError with comprehensive
        diagnostic and suggestion information.

        Args:
            target_norm (np.ndarray): The target objective point in normalized space.
            message (str): A user-friendly message explaining the failure.
            reason (FeasibilityFailureReason): The specific reason for the feasibility failure.
            score (float | None): The computed feasibility score, if applicable.
            extra_info (str | None): Additional diagnostic information.
            num_suggestions (int): The number of feasible suggestions to generate.

        Raises:
            ObjectiveOutOfBoundsError: Always raises this exception.
        """
        raise ObjectiveOutOfBoundsError(
            message=message,
            reason=reason,
            score=score,
            suggestions=self._generate_suggestions(target_norm, num_suggestions),
            extra_info=extra_info,
        )

    # -----------------------
    # Public API
    # -----------------------

    def validate(
        self,
        target: np.ndarray,
        target_norm: np.ndarray,
        num_suggestions: int = 3,
    ) -> None:
        """
        Validates the feasibility of a target objective point by performing
        a series of checks:
        1.  Checks if the target is within the raw historical objective bounds.
        2.  Calculates a feasibility score using the provided scoring strategy
            and compares it against the internal tolerance.

        If any check fails, an ObjectiveOutOfBoundsError is raised with
        details and suggestions.

        Args:
            target (np.ndarray): The target objective point in its original scale.
            target_norm (np.ndarray): The target objective point in normalized space.
            num_suggestions (int): The number of feasible suggestions to provide
                                   if an error is raised.

        Raises:
            ObjectiveOutOfBoundsError: If the target objective is deemed infeasible
                                       by any of the validation steps.
        """
        target = self._ensure_2d(target)
        target_norm = self._ensure_2d(target_norm)

        # Step 1: Raw bound checking - ensure target is within the overall range
        # of the historical Pareto front for each objective.
        violations = self._within_bounds(target)
        if violations:
            self._raise_error(
                target_norm=target_norm,
                message="Target is outside the bounds of the historical Pareto front.",
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                score=None,  # Score not applicable for raw bounds check
                extra_info="\n".join(violations),
                num_suggestions=num_suggestions,
            )

        # Step 2: Feasibility Score - evaluate how "close" or "dense" the target
        # is to the normalized Pareto front using the chosen scoring strategy.
        score = self._scorer.compute_score(target_norm, self._pareto_front_norm)

        # Check if the computed score indicates feasibility based on the scorer's logic
        # and the global tolerance.
        if not self._scorer.is_feasible(score, self._tolerance):
            self._raise_error(
                target_norm=target_norm,
                message="Target is too far from the Pareto front.",
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                score=score,
                extra_info=f"Computed score = {score:.16f}, tolerance = {self._tolerance:.4f}",
                num_suggestions=num_suggestions,
            )
