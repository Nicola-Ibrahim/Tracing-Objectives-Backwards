"""Type aliases shared across assurance domain modules."""

from typing import Protocol, Sequence

import numpy as np


class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray: ...


ArrayLike = np.ndarray | Sequence[float] | SupportsArray
Score = float
Vector = ArrayLike
Matrix = np.ndarray | Sequence[Sequence[float]] | SupportsArray


__all__ = ["ArrayLike", "Score", "Vector", "Matrix", "SupportsArray"]
