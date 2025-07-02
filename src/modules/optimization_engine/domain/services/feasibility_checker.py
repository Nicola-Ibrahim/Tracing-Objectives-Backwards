import numpy as np

from ..interpolation.exceptions import ObjectiveOutOfBoundsError


class ObjectiveFeasibilityChecker:
    """
    Checks if a target objective lies close enough to a known Pareto front.
    """

    def __init__(self, normalized_pareto_front: np.ndarray, tolerance: float):
        self._pareto_front = normalized_pareto_front
        self._tolerance = tolerance

    def get_distance(self, target: np.ndarray) -> float:
        return np.min(np.linalg.norm(self._pareto_front - target, axis=1))

    def is_feasible(self, target: np.ndarray) -> bool:
        return self.get_distance(target) <= self._tolerance

    def get_nearest_suggestions(self, target: np.ndarray, num: int) -> np.ndarray:
        distances = np.linalg.norm(self._pareto_front - target, axis=1)
        nearest_indices = np.argsort(distances)[:num]
        return self._pareto_front[nearest_indices]

    def validate(self, target: np.ndarray, num_suggestions: int = 3) -> None:
        """
        Raises an exception if the objective is too far from the front.
        """
        if not self.is_feasible(target):
            distance = self.get_distance(target)
            suggestions = self.get_nearest_suggestions(target, num_suggestions)
            raise ObjectiveOutOfBoundsError(distance, suggestions)
