"""Immutable representation of an objective vector in raw and normalised space."""

from dataclasses import dataclass

import numpy as np

from ...shared.ndarray_utils import ensure_2d


@dataclass(slots=True, frozen=True)
class ObjectiveVector:
    raw: np.ndarray
    normalized: np.ndarray

    def __post_init__(self) -> None:
        raw = ensure_2d(np.asarray(self.raw, dtype=float))
        normalized = ensure_2d(np.asarray(self.normalized, dtype=float))
        if raw.shape != normalized.shape:
            raise ValueError(
                "Raw and normalised objective vectors must share the same shape."
            )
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "normalized", normalized)

    @classmethod
    def from_raw(
        cls,
        raw: np.ndarray,
        *,
        normalizer,
    ) -> "ObjectiveVector":
        raw_2d = ensure_2d(np.asarray(raw, dtype=float))
        normalized = normalizer.transform(raw_2d)
        return cls(raw=raw_2d, normalized=normalized)


__all__ = ["ObjectiveVector"]
