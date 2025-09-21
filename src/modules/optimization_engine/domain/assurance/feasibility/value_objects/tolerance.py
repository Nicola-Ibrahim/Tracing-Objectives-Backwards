"""Tolerance configuration for feasibility evaluation."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class Tolerance:
    eps_l2: float | None = None
    eps_per_obj: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.eps_l2 is None and self.eps_per_obj is None:
            raise ValueError("Provide at least one tolerance (eps_l2 or eps_per_obj).")
        if self.eps_l2 is not None and self.eps_l2 < 0:
            raise ValueError("eps_l2 must be non-negative.")
        if self.eps_per_obj is not None:
            arr = np.asarray(self.eps_per_obj, dtype=float)
            if np.any(arr < 0):
                raise ValueError("eps_per_obj entries must be non-negative.")


__all__ = ["Tolerance"]
