from typing import Any, List

from pydantic import BaseModel


class AccuracySummary(BaseModel):
    mean_best_shot: float
    median_best_shot: float
    mean_bias: float
    mean_dispersion: float


class AccuracyLens(BaseModel):
    discrepancy_scores: Any  # np.ndarray
    best_shot_residuals: Any  # np.ndarray
    systematic_bias: Any  # np.ndarray
    cloud_dispersion: Any  # np.ndarray
    scenarios: List[str]
    summary: AccuracySummary

    class Config:
        arbitrary_types_allowed = True


class ReliabilitySummary(BaseModel):
    mean_crps: float
    mean_diversity: float
    mean_interval_width: float


class CalibrationCurve(BaseModel):
    pit_values: Any  # np.ndarray
    cdf_y: Any

    class Config:
        arbitrary_types_allowed = True


class ReliabilityLens(BaseModel):
    pit_values: Any  # np.ndarray
    calibration_error: float
    crps: float
    diversity: Any  # np.ndarray
    interval_width: Any  # np.ndarray
    summary: ReliabilitySummary
    calibration_curve: CalibrationCurve

    class Config:
        arbitrary_types_allowed = True


class SpatialCandidates(BaseModel):
    ordered: Any  # np.ndarray
    median: Any  # np.ndarray
    std: Any  # np.ndarray

    class Config:
        arbitrary_types_allowed = True


class DiagnosticResult(BaseModel):
    """
    Final Output Schema for Inverse Model Diagnostics.
    Enforces the Dual-Lens framework.
    """

    accuracy: AccuracyLens
    reliability: ReliabilityLens
    candidates: SpatialCandidates
