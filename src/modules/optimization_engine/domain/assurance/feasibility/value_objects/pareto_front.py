"""Lightweight wrapper around a Pareto front for invariant checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...shared.ndarray_utils import ensure_2d


@dataclass(slots=True, frozen=True)
class ParetoFront:
    raw: np.ndarray
    normalized: np.ndarray

    def __post_init__(self) -> None:
        if self.raw.shape != self.normalized.shape:
            raise ValueError("Raw and normalised Pareto fronts must share the same shape.")

    @property
    def objective_dim(self) -> int:
        return int(self.raw.shape[1])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.raw.min(axis=0), self.raw.max(axis=0)


__all__ = ["ParetoFront"]
