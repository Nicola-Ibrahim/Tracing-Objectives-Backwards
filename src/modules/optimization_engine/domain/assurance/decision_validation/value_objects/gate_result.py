from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: dict[str, float]
    explanation: str


__all__ = ["GateResult"]
