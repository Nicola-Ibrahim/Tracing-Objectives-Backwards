"""Scoring strategy port for feasibility assessments."""

from abc import ABC, abstractmethod

import numpy as np


class FeasibilityScoringStrategy(ABC):
    @abstractmethod
    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float:
        """Return a feasibility score for ``target`` against ``pareto_points``."""

    def is_feasible(self, score: float, threshold: float) -> bool:
        return score >= threshold


__all__ = ["FeasibilityScoringStrategy"]
