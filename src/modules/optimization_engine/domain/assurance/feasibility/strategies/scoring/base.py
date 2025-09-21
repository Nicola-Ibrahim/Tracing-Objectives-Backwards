"""Base classes for feasibility scoring strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeasibilityScoringStrategy(ABC):
    @abstractmethod
    def compute_score(self, target: np.ndarray, pareto_points: np.ndarray) -> float: ...

    def is_feasible(self, score: float, threshold: float) -> bool:
        return score >= threshold


__all__ = ["FeasibilityScoringStrategy"]
