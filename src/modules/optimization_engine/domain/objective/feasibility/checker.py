import warnings
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist  # For efficient pairwise distance calculation
from sklearn.cluster import KMeans

# Assuming FeasibilityFailureReason, ObjectiveOutOfBoundsError, FeasibilityScoringStrategy
# are imported from their respective modules
from .enums import FeasibilityFailureReason
from .exceptions import ObjectiveOutOfBoundsError
from .scoring_strategies.base import FeasibilityScoringStrategy


class ObjectiveFeasibilityChecker:
    """
    Domain service responsible for validating the feasibility of a given target
    objective against a historical Pareto front.
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
        Returns a list of violation messages for any out-of-bound features.
        An empty list means all features are within bounds.
        """
        target = self._ensure_2d(target)
        violations = []

        for i in range(target.shape[1]):
            val = target[0, i]
            if not (self._bounds_min[i] <= val <= self._bounds_max[i]):
                violations.append(
                    f"Feature {i}: {val:.4f} not in [{self._bounds_min[i]:.4f}, {self._bounds_max[i]:.4f}]"
                )
        return violations

    def _generate_suggestions(
        self,
        target_norm: np.ndarray,
        num_suggestions: int = 5,
        initial_pool_multiplier: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "max_min_distance",
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates diverse and close suggestions for feasible objective points
        from the historical Pareto front in normalized space.

        This method first identifies a larger pool of nearest neighbors (using Euclidean distance),
        then applies a diversity selection strategy to select `num_suggestions` points,
        and finally adds a slight perturbation for exploration.

        Args:
            target_norm (np.ndarray): The infeasible target objective point in
                                      normalized space (e.g., [1, D]).
            num_suggestions (int): The desired number of diverse suggestions to return.
            initial_pool_multiplier (int): Multiplier for the initial pool size of
                                           nearest Pareto points to consider. The pool
                                           size will be `initial_pool_multiplier * num_suggestions`.
                                           A larger multiplier increases the chance of finding
                                           diverse points but may increase computation.
            suggestion_noise_scale (float): The maximum relative magnitude of random
                                            perturbation applied to each suggestion.
                                            Noise will be in range +/- (`_tolerance` * `suggestion_noise_scale`).
                                            Should be between 0.0 and 1.0.
            diversity_method (str): Method to select diverse points from the nearest pool:
                                    - "max_min_distance": Selects points iteratively to maximize the
                                                          minimum distance between selected points.
                                    - "kmeans": Uses K-Means clustering to find `num_suggestions`
                                                representative points from the pool. Requires scikit-learn.
                                    - "none": Simply returns the `num_suggestions` closest points
                                              from the `initial_pool_multiplier` nearest.
            random_seed (Optional[int]): Seed for the random number generator to ensure
                                         reproducibility of suggestions.

        Returns:
            np.ndarray: A 2D array of suggested feasible objective points in
                        normalized space, clipped to [0, 1]. Returns an empty array
                        if no Pareto points are available.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            # Set random state for KMeans if available and random_seed is provided
            KMeans_random_state = random_seed
        else:
            KMeans_random_state = None

        target_norm = self._ensure_2d(target_norm)

        if self._pareto_front_norm.shape[0] == 0:
            return np.array([])  # No Pareto points to generate suggestions from

        # Step 1: Get an initial pool of nearest Pareto points using Euclidean distance.
        pool_size = min(
            self._pareto_front_norm.shape[0],
            max(num_suggestions, initial_pool_multiplier * num_suggestions),
        )

        distances = np.linalg.norm(self._pareto_front_norm - target_norm, axis=1)
        nearest_pool_indices = np.argsort(distances)[:pool_size]
        nearest_pool = self._pareto_front_norm[nearest_pool_indices]

        selected_suggestions = []

        if nearest_pool.shape[0] == 0:
            return np.array([])  # No points in the pool, cannot generate suggestions

        if num_suggestions == 0:
            return np.array([])  # No suggestions requested

        # Step 2: Apply diversity selection from the nearest pool
        if diversity_method == "max_min_distance":
            if nearest_pool.shape[0] <= num_suggestions:
                # If the pool is smaller than or equal to the desired number, take all.
                selected_suggestions = nearest_pool
            else:
                # Start with the point in the pool closest to the target.
                pool_distances_to_target = np.linalg.norm(
                    nearest_pool - target_norm, axis=1
                )
                first_idx_in_pool = np.argmin(pool_distances_to_target)

                selected_suggestions.append(nearest_pool[first_idx_in_pool])
                remaining_indices = np.delete(
                    np.arange(nearest_pool.shape[0]), first_idx_in_pool
                )

                for _ in range(num_suggestions - 1):
                    if len(remaining_indices) == 0:
                        break  # No more points available to select

                    distances_to_selected = cdist(
                        nearest_pool[remaining_indices], np.array(selected_suggestions)
                    )
                    min_distances_per_remaining = np.min(distances_to_selected, axis=1)

                    next_idx_in_remaining = np.argmax(min_distances_per_remaining)
                    next_point_original_idx = remaining_indices[next_idx_in_remaining]

                    selected_suggestions.append(nearest_pool[next_point_original_idx])
                    remaining_indices = np.delete(
                        remaining_indices, next_idx_in_remaining
                    )

                selected_suggestions = np.array(selected_suggestions)

        elif diversity_method == "kmeans":
            if KMeans is None:
                warnings.warn(
                    "K-Means requires scikit-learn to be installed. Falling back to 'none' method."
                )
                selected_suggestions = nearest_pool[:num_suggestions]
            elif nearest_pool.shape[0] < num_suggestions:
                warnings.warn(
                    f"Not enough points in the nearest pool ({nearest_pool.shape[0]}) for K-Means with {num_suggestions} clusters. Falling back to 'none' method."
                )
                selected_suggestions = (
                    nearest_pool  # Take all available if less than num_suggestions
                )
            elif num_suggestions == 1:
                # K-Means with 1 cluster is trivial, just pick the closest point from the pool to the target
                pool_distances_to_target = np.linalg.norm(
                    nearest_pool - target_norm, axis=1
                )
                selected_suggestions = nearest_pool[
                    np.argmin(pool_distances_to_target)
                ].reshape(1, -1)
            else:
                # Initialize KMeans
                kmeans = KMeans(
                    n_clusters=num_suggestions,
                    random_state=KMeans_random_state,
                    n_init="auto",
                )
                kmeans.fit(nearest_pool)

                # Get cluster centroids
                cluster_centroids = kmeans.cluster_centers_

                # For each centroid, find the closest actual point from the nearest_pool
                # This ensures suggestions are real Pareto points (before perturbation)
                for centroid in cluster_centroids:
                    distances_to_centroid = np.linalg.norm(
                        nearest_pool - centroid, axis=1
                    )
                    closest_point_idx = np.argmin(distances_to_centroid)
                    selected_suggestions.append(nearest_pool[closest_point_idx])
                selected_suggestions = np.array(selected_suggestions)

        elif diversity_method == "none":
            # Just take the top `num_suggestions` closest points from the pool
            selected_suggestions = nearest_pool[:num_suggestions]

        else:  # Handle unknown diversity method
            warnings.warn(
                f"Unknown diversity method '{diversity_method}'. Falling back to 'none' method."
            )
            selected_suggestions = nearest_pool[:num_suggestions]

        # Final check if selected_suggestions is empty, despite pool having points.
        if (
            selected_suggestions.shape[0] == 0
            and nearest_pool.shape[0] > 0
            and num_suggestions > 0
        ):
            # As a robust fallback, pick random points from the original Pareto front.
            num_fallback = min(num_suggestions, self._pareto_front_norm.shape[0])
            random_indices = np.random.choice(
                self._pareto_front_norm.shape[0], num_fallback, replace=False
            )
            selected_suggestions = self._pareto_front_norm[random_indices]
            warnings.warn(
                "Fallback: No suggestions selected by diversity method. Picking random Pareto points."
            )
        elif selected_suggestions.shape[0] == 0:
            return np.array([])  # Still no suggestions, return empty

        # Step 3: Add slight random perturbation to the selected suggestions
        perturbation_range = self._tolerance * suggestion_noise_scale
        noise = np.random.uniform(
            -perturbation_range, perturbation_range, size=selected_suggestions.shape
        )
        perturbed_suggestions = selected_suggestions + noise

        # Step 4: Clip to ensure suggestions stay within the normalized [0, 1] range.
        return np.clip(perturbed_suggestions, 0.0, 1.0)

    def _raise_error(
        self,
        *,
        target_norm: np.ndarray,
        message: str,
        reason: FeasibilityFailureReason,
        score: float | None = None,
        extra_info: str | None = None,
        suggestions: np.ndarray,
    ) -> None:
        """
        Raises an ObjectiveOutOfBoundsError with comprehensive diagnostic information
        and recovery suggestions.

        Args:
            target_norm (np.ndarray): The target objective point in normalized space.
            message (str): A user-friendly message explaining the failure.
            reason (FeasibilityFailureReason): The specific reason for the feasibility failure.
            score (float | None): The computed feasibility score, if applicable.
            extra_info (str | None): Additional diagnostic information.
            suggestions (np.ndarray): Pre-computed array of suggested feasible objective points.

        Raises:
            ObjectiveOutOfBoundsError: Always raises this exception.
        """
        raise ObjectiveOutOfBoundsError(
            message=message,
            reason=reason,
            score=score,
            suggestions=suggestions,
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
        initial_pool_multiplier: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "max_min_distance",
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Validates the feasibility of a normalized target point against the historical Pareto front.

        It performs two main checks:
        1.  **Raw Bounds Check:** Verifies if the target objective's values are within the
            minimum and maximum observed values of the historical Pareto front in original space.
            This ensures the target is within the overall range of observed objectives.
        2.  **Feasibility Score Check:** Utilizes a specified `FeasibilityScoringStrategy`
            to calculate a score indicating how "close" or "dense" the target is to the
            normalized Pareto front. This score is then compared against a predefined
            `_tolerance` threshold.

        If either check fails, an `ObjectiveOutOfBoundsError` is raised. This exception
        includes detailed diagnostic information about the failure and a set of
        suggested feasible objective points derived from the Pareto front. Suggestions are
        only generated if a violation occurs.

        Args:
            target (np.ndarray): The target objective point in its original, unnormalized scale.
                                 Expected shape (D,) or (1, D).
            target_norm (np.ndarray): The target objective point in normalized space (e.g., [0, 1] per objective).
                                      Expected shape (D,) or (1, D).
            num_suggestions (int): The number of diverse feasible suggestions to generate
                                   and include in the raised error, if applicable.
            initial_pool_multiplier (int): Controls the pool size for suggestion generation.
            suggestion_noise_scale (float): Controls the magnitude of random noise for suggestions.
            diversity_method (str): Specifies the method for selecting diverse suggestions.
            random_seed (Optional[int]): Seed for reproducibility of random operations in suggestions.

        Raises:
            ObjectiveOutOfBoundsError: If the target is found to be outside raw bounds
                                       or too far from the Pareto front based on the score.
        """
        target = self._ensure_2d(target)
        target_norm = self._ensure_2d(target_norm)

        # Step 1: Raw bound checking
        violations = self._within_bounds(target)
        if violations:
            # ONLY generate suggestions if there's an actual violation in raw bounds
            suggestions = self._generate_suggestions(
                target_norm=target_norm,
                num_suggestions=num_suggestions,
                initial_pool_multiplier=initial_pool_multiplier,
                suggestion_noise_scale=suggestion_noise_scale,
                diversity_method=diversity_method,
                random_seed=random_seed,
            )
            self._raise_error(
                target_norm=target_norm,
                message="Target is outside the bounds of the historical Pareto front.",
                reason=FeasibilityFailureReason.OUT_OF_RAW_BOUNDS,
                score=None,
                extra_info="\n".join(violations),
                suggestions=suggestions,
            )

        # Step 2: Feasibility Score
        score = self._scorer.compute_score(target_norm, self._pareto_front_norm)
        # print(score) # Commented out, typically for debugging

        if not self._scorer.is_feasible(score, self._tolerance):
            # ONLY generate suggestions if there's an actual violation in feasibility score
            suggestions = self._generate_suggestions(
                target_norm=target_norm,
                num_suggestions=num_suggestions,
                initial_pool_multiplier=initial_pool_multiplier,
                suggestion_noise_scale=suggestion_noise_scale,
                diversity_method=diversity_method,
                random_seed=random_seed,
            )
            self._raise_error(
                target_norm=target_norm,
                message="Target is too far from the Pareto front.",
                reason=FeasibilityFailureReason.TOO_FAR_FROM_FRONT,
                score=score,
                extra_info=f"Computed score = {score:.16f}, tolerance = {self._tolerance:.4f}",
                suggestions=suggestions,
            )
