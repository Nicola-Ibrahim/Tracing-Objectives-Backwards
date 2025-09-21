"""Shared calibration value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class OODCalibration:
    mu: np.ndarray
    prec: np.ndarray
    threshold_md2: float


@dataclass(slots=True, frozen=True)
class ConformalCalibration:
    radius_q: float


__all__ = ["OODCalibration", "ConformalCalibration"]
