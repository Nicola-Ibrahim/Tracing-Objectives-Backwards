"""Diversity strategy port for generating feasible suggestions."""

from abc import ABC, abstractmethod

import numpy as np


class BaseDiversityStrategy(ABC):
    def __init__(self, random_seed: int | None = None) -> None:
        self._random_seed = random_seed

    @abstractmethod
    def select_diverse_points(
        self,
        *,
        pareto_front_normalized: np.ndarray,
        target_normalized: np.ndarray,
        num_suggestions: int,
    ) -> np.ndarray: ...


__all__ = ["BaseDiversityStrategy"]
