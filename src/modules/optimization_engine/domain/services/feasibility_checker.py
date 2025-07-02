import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from ..interpolation.exceptions import (
    FeasibilityFailureReason,
    ObjectiveOutOfBoundsError,
)


class ObjectiveFeasibilityChecker:
    """
    Checks whether a target objective is within the feasible (convex) region
    of the Pareto front in objective space, and guides the user accordingly.
    All internal checks and returned suggestions are based on NORMALIZED data.
    """

    def __init__(
        self,
        pareto_front: np.ndarray,  # Unnormalized Pareto front (for bounds check)
        pareto_front_norm: np.ndarray,  # Normalized Pareto front (for hull/distance)
        tolerance: float,  # Tolerance for distance to the front (after being inside hull)
    ):
        if pareto_front.shape[1] != pareto_front_norm.shape[1]:
            raise ValueError(
                "Unnormalized and normalized Pareto fronts must have the same number of features."
            )
        if pareto_front.shape[0] < pareto_front.shape[1] + 1:
            # For ConvexHull, need at least D+1 points for D dimensions
            raise ValueError(
                f"Not enough points in Pareto front ({pareto_front.shape[0]}) to form a convex hull "
                f"in {pareto_front.shape[1]} dimensions. Need at least {pareto_front.shape[1] + 1} points."
            )

        self._pareto_front = pareto_front
        self._pareto_front_norm = pareto_front_norm
        self._tolerance = tolerance

        self._hull: ConvexHull | None = None
        self._delaunay: Delaunay | None = None
        self._min_unnorm_bounds: np.ndarray | None = None
        self._max_unnorm_bounds: np.ndarray | None = None

        self._precompute_geometry()

    def _precompute_geometry(self):
        """Precomputes bounding box, convex hull, and Delaunay triangulation for efficiency."""
        self._min_unnorm_bounds = self._pareto_front.min(axis=0)
        self._max_unnorm_bounds = self._pareto_front.max(axis=0)
        try:
            self._hull = ConvexHull(self._pareto_front_norm)
            self._delaunay = Delaunay(self._pareto_front_norm)
        except Exception as e:
            # Handle cases where hull/delaunay computation might fail (e.g., all points collinear, not enough points)
            raise RuntimeError(
                f"Failed to compute geometric structures for Pareto front: {e}"
            )

    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        """Helper to ensure array is 2D."""
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def _euclidean_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculates Euclidean distance between each row of A and each row of B."""
        return np.linalg.norm(A - B, axis=1)

    def _is_inside_convex_hull(
        self, target_norm: np.ndarray, tol_check: float = 1e-9
    ) -> bool:
        """
        Checks if a normalized target point lies inside the precomputed convex hull.
        Uses Delaunay triangulation (more robust for "inside" check than hull equations alone).
        """
        if self._delaunay is None:
            raise RuntimeError("Delaunay triangulation not precomputed.")
        target_norm = self._reshape(target_norm)
        # find_simplex returns -1 if point is outside any simplex (i.e., outside the convex hull)
        return np.all(self._delaunay.find_simplex(target_norm) >= 0)

    def _get_distance_to_nearest_pareto_point(self, target_norm: np.ndarray) -> float:
        """
        Calculates the minimum Euclidean distance from a normalized target
        to any point in the normalized Pareto front.
        """
        target_norm = self._reshape(target_norm)
        return float(
            np.min(self._euclidean_distance(self._pareto_front_norm, target_norm))
        )

    def get_nearest_suggestions(self, target_norm: np.ndarray, num: int) -> np.ndarray:
        """
        Returns the 'num' nearest points from the normalized Pareto front to the target.
        These are returned in normalized space.
        """
        if num <= 0:
            return np.array([])
        target_norm = self._reshape(target_norm)
        distances = self._euclidean_distance(self._pareto_front_norm, target_norm)
        nearest_indices = np.argsort(distances)[:num]
        return self._pareto_front_norm[nearest_indices]

    def _check_within_unnormalized_objective_bounds(self, target: np.ndarray) -> None:
        """
        Ensures the raw (unnormalized) target is within min/max bounds of the original Pareto front.
        Raises ValueError with detailed violations if not.
        """
        if self._min_unnorm_bounds is None or self._max_unnorm_bounds is None:
            raise RuntimeError("Unnormalized bounds not precomputed.")

        target = self._reshape(target)

        violations = []
        for i in range(target.shape[1]):
            is_below = target[0, i] < self._min_unnorm_bounds[i]
            is_above = target[0, i] > self._max_unnorm_bounds[i]
            if is_below or is_above:
                violations.append(
                    f"Feature {i}: value = {target[0, i]:.4f}, "
                    f"allowed range = [{self._min_unnorm_bounds[i]:.4f}, {self._max_unnorm_bounds[i]:.4f}]"
                )

        if violations:
            # Raise a standard ValueError here, which `validate` will catch and wrap
            raise ValueError(
                "Target objective out of raw bounds:\n" + "\n".join(violations)
            )

    def validate(
        self,
        target: np.ndarray,  # Unnormalized target objective for initial bounds check
        target_norm: np.ndarray,  # Normalized target objective for hull/distance checks
        num_suggestions: int = 3,
    ) -> None:
        """
        Validates whether a normalized target objective is feasible.
        - Within the original (unnormalized) Pareto front's bounding box.
        - Inside the convex hull of the normalized Pareto front.
        - Within a specified tolerance of any point on the normalized Pareto front.

        Raises ObjectiveOutOfBoundsError with specific messages and reasons
        depending on which validation step fails.
        """
        # Ensure precomputation was successful
        if (
            self._hull is None
            or self._delaunay is None
            or self._min_unnorm_bounds is None
        ):
            raise RuntimeError(
                "Feasibility checker not fully initialized. Geometry or bounds missing."
            )

        target = self._reshape(target)
        target_norm = self._reshape(target_norm)

        # 1. Check if unnormalized target is within the overall bounding box
        try:
            self._check_within_unnormalized_objective_bounds(target)
        except ValueError as e:  # Catch the specific ValueError from _check_within_unnormalized_objective_bounds
            raise ObjectiveOutOfBoundsError(
                message="Target objective is outside the original Pareto front's historical range.",
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                distance=float("inf"),  # Indicate a severe, unquantifiable distance
                suggestions=self.get_nearest_suggestions(
                    target_norm=target_norm, num=num_suggestions
                ),
                extra_info=str(e),  # Pass the detailed bounds error message
            )

        # 2. Check if normalized target is inside the convex hull
        if not self._is_inside_convex_hull(target_norm):
            distance_to_front = self._get_distance_to_nearest_pareto_point(target_norm)
            suggestions = self.get_nearest_suggestions(target_norm, num_suggestions)
            raise ObjectiveOutOfBoundsError(
                message="Target objective is outside the feasible region (convex hull) of the normalized Pareto front.",
                reason=FeasibilityFailureReason.OUTSIDE_CONVEX_HULL,
                distance=distance_to_front,
                suggestions=suggestions,
                extra_info=(
                    "This means there are no known solutions that can achieve this objective. "
                    "Consider choosing an objective within the range of past optimal designs."
                ),
            )

        # 3. Check if inside convex hull but too far from actual points on the front
        distance_to_front = self._get_distance_to_nearest_pareto_point(target_norm)
        if distance_to_front > self._tolerance:
            suggestions = self.get_nearest_suggestions(target_norm, num_suggestions)
            raise ObjectiveOutOfBoundsError(
                message="Target objective is within the feasible region but not close enough to any known optimal solution.",
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                distance=distance_to_front,
                suggestions=suggestions,
                extra_info=(
                    f"Minimum distance to an optimal solution: {distance_to_front:.4f} "
                    f"(tolerance: {self._tolerance:.4f}). "
                    "Adjust your objective slightly to be closer to existing optimal designs."
                ),
            )
