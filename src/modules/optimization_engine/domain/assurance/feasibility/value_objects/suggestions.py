"""Container for feasible objective suggestions."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class Suggestions:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Suggestions must be a 2-D array.")

    @property
    def count(self) -> int:
        return int(self.values.shape[0])


__all__ = ["Suggestions"]
