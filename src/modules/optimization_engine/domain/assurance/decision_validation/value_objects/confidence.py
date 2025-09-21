from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ConfidenceLevel:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 < self.value < 1.0):
            raise ValueError("confidence value must lie in (0, 1)")


__all__ = ["ConfidenceLevel"]
