from abc import ABC, abstractmethod

import numpy as np


class FeasibilityScoringStrategy(ABC):
    """
    Base class for feasibility scoring strategies.
    """

    @abstractmethod
    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """
        Compute a feasibility score in [0, 1] based on target and Pareto front.
        Higher = more feasible.
        """
        pass

    def is_feasible(self, score: float, threshold: float) -> bool:
        return score >= threshold
