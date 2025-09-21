"""Immutable representation of an objective vector in raw and normalised space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...shared.ndarray_utils import ensure_2d


@dataclass(slots=True, frozen=True)
class ObjectiveVector:
    raw: np.ndarray
    normalized: np.ndarray

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
