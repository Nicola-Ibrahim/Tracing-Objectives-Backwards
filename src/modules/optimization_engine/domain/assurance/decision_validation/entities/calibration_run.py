"""Entity representing a calibration execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CalibrationRun:
    X_cal: np.ndarray
    y_cal: np.ndarray
    notes: dict[str, float] | None = None


__all__ = ["CalibrationRun"]
