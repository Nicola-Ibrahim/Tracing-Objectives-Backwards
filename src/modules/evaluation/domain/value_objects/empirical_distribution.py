from typing import Self

import numpy as np
from pydantic import BaseModel


class EmpiricalDistribution(BaseModel):
    """
    Value Object: Empirical Cumulative Distribution Function (ECDF).

    A domain-standard way to represent the distribution of any
    diagnostic metric (discrepancy scores, PIT values, etc.).
    """

    x: list[float]  # sorted values
    y: list[float]  # cumulative probabilities

    @classmethod
    def from_samples(cls, samples: np.ndarray, max_points: int = 100) -> Self:
        """Factory: builds an ECDF from raw sample values."""
        arr = np.asarray(samples).flatten()
        if arr.size == 0:
            return cls(x=[], y=[])

        x = np.sort(arr)
        y = np.arange(1, len(x) + 1) / len(x)

        # Resample to max_points if too large for bandwidth efficiency
        if len(x) > max_points:
            indices = np.linspace(0, len(x) - 1, max_points).astype(int)
            x = x[indices]
            y = y[indices]

        return cls(x=x.tolist(), y=y.tolist())
