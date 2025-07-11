import warnings

import numpy as np

from .diversities import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
)
from .exceptions import ObjectiveOutOfBoundsError
from .scorers.base import FeasibilityScoringStrategy
from .validators import (
    BaseFeasibilityValidator,
    HistoricalObjectiveRangeValidator,
    ParetoFrontProximityValidator,
)


class ObjectiveFeasibilityChecker:
    """
    Domain service responsible for validating the feasibility of a given candidate
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

        self._min_historical_objectives = pareto_front.min(axis=0)
        self._max_historical_objectives = pareto_front.max(axis=0)

        # Mapping for diversity strategies
        self._diversity_strategies_map = {
            "max_min_distance": MaxMinDistanceDiversityStrategy,
            "kmeans": KMeansDiversityStrategy,
            "none": ClosestPointsDiversityStrategy,
        }

    # -----------------------
    # Utils
    # -----------------------
    def _ensure_2d(self, arr: np.ndarray) -> np.ndarray:
        """Ensures an array is 2-dimensional (row vector if 1D)."""
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def _generate_suggestions(
        self,
        target_normalized: np.ndarray,
        num_suggestions: int = 3,
        initial_pool_multiplier: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "max_min_distance",
        random_seed: int = None,  # Corrected type hint
    ) -> np.ndarray:
        """
        Generates diverse and close suggestions for feasible objective points
        from the historical Pareto front in normalized space using a specified
        diversity strategy.

        Args:
            target_normalized (np.ndarray): The infeasible candidate objective point in
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
        np.random.seed(44)

        target_normalized = self._ensure_2d(target_normalized)

        if self._pareto_front_norm.shape[0] == 0:
            return np.array([])

        pool_size = min(
            self._pareto_front_norm.shape[0],
            max(num_suggestions, initial_pool_multiplier * num_suggestions),
        )

        # Calculate distances to target and get the nearest pool points
        distances = np.linalg.norm(self._pareto_front_norm - target_normalized, axis=1)
        nearest_pool_indices = np.argsort(distances)[:pool_size]
        nearest_pool = self._pareto_front_norm[nearest_pool_indices]

        if nearest_pool.shape[0] == 0 or num_suggestions == 0:
            return np.array([])

        # Select the diversity strategy
        diversity_strategy_cls = self._diversity_strategies_map.get(
            diversity_method, ClosestPointsDiversityStrategy
        )

        # Instantiate the strategy and select points
        diversity_strategy = diversity_strategy_cls(random_seed=random_seed)
        selected_suggestions = diversity_strategy.select_diverse_points(
            pool_points=nearest_pool,
            num_desired_points=num_suggestions,
            target_normalized=target_normalized,
        )

        # Fallback if no suggestions are returned by the strategy but points exist
        if (
            selected_suggestions.shape[0] == 0
            and nearest_pool.shape[0] > 0
            and num_suggestions > 0
        ):
            warnings.warn(
                "Fallback: No suggestions selected by diversity strategy. Picking random Pareto points."
            )
            num_fallback = min(num_suggestions, self._pareto_front_norm.shape[0])
            random_indices = np.random.choice(
                self._pareto_front_norm.shape[0], num_fallback, replace=False
            )
            selected_suggestions = self._pareto_front_norm[random_indices]

        # Apply noise and clip
        perturbation_range = self._tolerance * suggestion_noise_scale
        noise = np.random.uniform(
            -perturbation_range, perturbation_range, size=selected_suggestions.shape
        )
        perturbed_suggestions = selected_suggestions + noise

        return np.clip(perturbed_suggestions, 0.0, 1.0)

    def _validate_by(
        self,
        *validators: BaseFeasibilityValidator,  # Accepts multiple validators as positional arguments
        target_normalized: np.ndarray,
        num_suggestions: int,
        initial_pool_multiplier: int,
        suggestion_noise_scale: float,
        diversity_method: str,
    ) -> None:
        """
        Executes a series of validators. Raises an error if any validator finds a violation,
        stopping the process immediately. This method is analogous to the 'self.check_rules'
        in your example.
        """
        for validator in validators:
            validation_result = validator.validate()

            if not validation_result.is_feasible:
                # If a validator fails, immediately generate suggestions and raise the error.
                suggestions = self._generate_suggestions(
                    target_normalized=target_normalized,
                    num_suggestions=num_suggestions,
                    initial_pool_multiplier=initial_pool_multiplier,
                    suggestion_noise_scale=suggestion_noise_scale,
                    diversity_method=diversity_method,
                )

                # with all the necessary parameters.
                raise ObjectiveOutOfBoundsError(
                    message=f"Validation failed: {validation_result.reason.value}"
                    if validation_result.reason
                    else "Validation failed.",
                    reason=validation_result.reason,
                    score=validation_result.score,
                    suggestions=suggestions,
                    extra_info=validation_result.extra_info,
                )

                # The error is raised, so the function will exit here, breaking out of the loop implicitly.
        # If the loop completes without an error, all validators passed.

    # -----------------------
    # Public API
    # -----------------------

    def validate(
        self,
        target: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int = 3,
        initial_pool_multiplier: int = 3,
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "max_min_distance",
    ) -> None:
        """
        Validates the feasibility of a candidate objective point against the historical Pareto front.

        It performs checks by instantiating and executing specific validators. If any validator
        determines the candidate is not feasible, an `ObjectiveOutOfBoundsError` is raised.
        This exception includes detailed diagnostic information about the failure and a set of
        suggested feasible objective points derived from the Pareto front. Suggestions are
        only generated if a violation occurs.

        Args:
            target (np.ndarray): The candidate objective point in its original, unnormalized scale.
                                 Expected shape (D,) or (1, D).
            target_normalized (np.ndarray): The candidate objective point in normalized space (e.g., [0, 1] per objective).
                                      Expected shape (D,) or (1, D).
            num_suggestions (int): The number of diverse feasible suggestions to generate
                                   and include in the raised error, if applicable.
            initial_pool_multiplier (int): Controls the pool size for suggestion generation.
            suggestion_noise_scale (float): Controls the magnitude of random noise for suggestions.
            diversity_method (str): Specifies the method for selecting diverse suggestions.
            random_seed (int): Seed for reproducibility of random operations in suggestions.

        Raises:
            ObjectiveOutOfBoundsError: If the candidate is found to be outside historical objective range
                                       or too far from the Pareto front based on the score.
        """
        target = self._ensure_2d(target)
        target_normalized = self._ensure_2d(target_normalized)

        self._validate_by(
            HistoricalObjectiveRangeValidator(
                target=target,
                historical_min_values=self._min_historical_objectives,
                historical_max_values=self._max_historical_objectives,
            ),
            ParetoFrontProximityValidator(
                target_normalized=target_normalized,
                scorer=self._scorer,
                tolerance=self._tolerance,
                historical_normalized_front=self._pareto_front_norm,
            ),
            target_normalized=target_normalized,
            num_suggestions=num_suggestions,
            initial_pool_multiplier=initial_pool_multiplier,
            suggestion_noise_scale=suggestion_noise_scale,
            diversity_method=diversity_method,
        )

        # If all validators pass, the method completes without raising an error.
