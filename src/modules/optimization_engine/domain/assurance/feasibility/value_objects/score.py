"""Feasibility score value object."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Score:
    value: float

    def __post_init__(self) -> None:
        if not (self.value >= 0.0):
            raise ValueError("Score must be non-negative.")


__all__ = ["Score"]
