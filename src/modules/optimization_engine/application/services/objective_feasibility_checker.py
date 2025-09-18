import warnings

import numpy as np

from ...domain.assurance.services.feasibility.diversities import (
    ClosestPointsDiversityStrategy,
    KMeansDiversityStrategy,
    MaxMinDistanceDiversityStrategy,
)
from ...domain.assurance.services.feasibility.exceptions import ObjectiveOutOfBoundsError
from ...domain.assurance.services.feasibility.scorers.base import FeasibilityScoringStrategy
from ...domain.assurance.services.feasibility.validators import (
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
        pareto_front_normalized: np.ndarray,
        tolerance: float,
        scorer: FeasibilityScoringStrategy,
    ):
        """
        Initializes the ObjectiveFeasibilityChecker.

        Args:
            pareto_front (np.ndarray): The historical Pareto front in its original,
                                       unnormalized objective space. Expected shape (N, D),
                                       where N is the number of points and D is the number of objectives.
            pareto_front_normalized (np.ndarray): The historical Pareto front, normalized to a
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
        if pareto_front.shape[1] != pareto_front_normalized.shape[1]:
            raise ValueError(
                "Pareto front and normalized front must have same dimensionality."
            )

        self._pareto_front = pareto_front
        self._pareto_front_normalized = pareto_front_normalized
        self._tolerance = tolerance
        self._scorer = scorer

        self._min_historical_objectives = pareto_front.min(axis=0)
        self._max_historical_objectives = pareto_front.max(axis=0)

        # Mapping for diversity strategies
        self._diversity_strategies_map = {
            "max_min_distance": MaxMinDistanceDiversityStrategy,
            "kmeans": KMeansDiversityStrategy,
            "euclidean": ClosestPointsDiversityStrategy,
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
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "euclidean",
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Generates diverse and close suggestions for feasible objective points
        from the historical Pareto front in normalized space using a specified
        diversity strategy. The responsibility of identifying relevant points
        (e.g., finding closest points or applying clustering) is delegated
        entirely to the chosen strategy.

        Args:
            target_normalized (np.ndarray): The infeasible candidate objective point in
                                      normalized space (e.g., [1, D]).
            num_suggestions (int): The desired number of diverse suggestions to return.
            suggestion_noise_scale (float): The maximum relative magnitude of random
                                            perturbation applied to each suggestion.
                                            Noise will be in range +/- (`_tolerance` * `suggestion_noise_scale`).
                                            Should be between 0.0 and 1.0.
            diversity_method (str): Method to select diverse points from the Pareto front:
                                    - "max_min_distance": Selects points iteratively to maximize the
                                                          minimum distance between selected points from the full front.
                                    - "kmeans": Uses K-Means clustering to find `num_suggestions`
                                                representative points from the full front. Requires scikit-learn.
                                    - "euclidean": Simply returns the `num_suggestions` closest points
                                              from the entire front.
            random_seed (Optional[int]): Seed for the random number generator to ensure
                                         reproducibility of suggestions.

        Returns:
            np.ndarray: A 2D array of suggested feasible objective points in
                        normalized space, clipped to [0, 1]. Returns an empty array
                        if no Pareto points are available.
        """
        target_normalized = self._ensure_2d(target_normalized)

        if self._pareto_front_normalized.shape[0] == 0 or num_suggestions == 0:
            return np.array([])

        # Select the diversity strategy
        diversity_strategy_cls = self._diversity_strategies_map.get(
            diversity_method, ClosestPointsDiversityStrategy
        )
        # Instantiate the strategy and select points
        diversity_strategy = diversity_strategy_cls(random_seed=random_seed)
        selected_suggestions = diversity_strategy.select_diverse_points(
            pareto_front_normalized=self._pareto_front_normalized,
            target_normalized=target_normalized,
            num_suggestions=num_suggestions,
        )

        # Fallback if no suggestions are returned by the strategy but points exist
        if (
            selected_suggestions.shape[0] == 0
            and self._pareto_front_normalized.shape[0] > 0
            and num_suggestions > 0
        ):
            warnings.warn(
                "Fallback: No suggestions selected by diversity strategy. Picking random Pareto points."
            )
            if random_seed is not None:
                np.random.seed(random_seed)  # Ensure reproducibility for fallback
            num_fallback = min(num_suggestions, self._pareto_front_normalized.shape[0])
            random_indices = np.random.choice(
                self._pareto_front_normalized.shape[0], num_fallback, replace=False
            )
            selected_suggestions = self._pareto_front_normalized[random_indices]

        # Apply noise and clip
        if random_seed is not None:
            np.random.seed(random_seed)  # Ensure reproducibility for noise
        perturbation_range = self._tolerance * suggestion_noise_scale
        noise = np.random.uniform(
            -perturbation_range, perturbation_range, size=selected_suggestions.shape
        )
        perturbed_suggestions = selected_suggestions + noise

        return np.clip(perturbed_suggestions, 0.0, 1.0)

    def _validate_by(
        self,
        *validators: BaseFeasibilityValidator,
        target_normalized: np.ndarray,
        num_suggestions: int,
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
        suggestion_noise_scale: float = 0.05,
        diversity_method: str = "equlidean",
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
            suggestion_noise_scale (float): Controls the magnitude of random noise for suggestions.
            diversity_method (str): Specifies the method for selecting diverse suggestions.

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
                historical_normalized_front=self._pareto_front_normalized,
            ),
            target_normalized=target_normalized,
            num_suggestions=num_suggestions,
            suggestion_noise_scale=suggestion_noise_scale,
            diversity_method=diversity_method,
        )

        # If all validators pass, the method completes without raising an error.
