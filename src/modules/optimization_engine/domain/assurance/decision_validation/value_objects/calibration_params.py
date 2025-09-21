from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class OODCalibrationParams:
    percentile: float = 97.5
    cov_reg: float = 1e-6

    def __post_init__(self) -> None:
        if not (0 < self.percentile <= 100):
            raise ValueError("percentile must be in (0, 100]")
        if self.cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")


__all__ = ["OODCalibrationParams"]
