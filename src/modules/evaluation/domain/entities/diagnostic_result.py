from datetime import datetime
from typing import Any, List, Self
from uuid import uuid4

from pydantic import BaseModel, Field


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


class DiagnosticRunMetadata(BaseModel):
    """Metadata for a specific diagnostic evaluation run."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this specific evaluation run.",
    )
    run_number: int | None = Field(
        default=None,
        description="Sequential run number (e.g., 1, 2, 3) assigned by storage.",
    )
    estimator_type: str
    estimator_version: int
    dataset_name: str
    mapping_direction: str = "inverse"
    created_at: datetime = Field(default_factory=datetime.now)
    num_samples: int
    scale_method: str = "sd"


class DiagnosticResult(BaseModel):
    """
    Aggregate Root for Inverse Model Diagnostics.
    Enforces the Dual-Lens framework with serialization support.
    """

    metadata: DiagnosticRunMetadata
    accuracy: AccuracyLens
    reliability: ReliabilityLens
    candidates: SpatialCandidates

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        json_decoders = {datetime: datetime.fromisoformat}

    @classmethod
    def create(
        cls,
        estimator_type: str,
        estimator_version: int,
        dataset_name: str,
        num_samples: int,
        scale_method: str,
        accuracy: AccuracyLens,
        reliability: ReliabilityLens,
        candidates: SpatialCandidates,
        mapping_direction: str = "inverse",
    ) -> Self:
        """
        Factory for new diagnostic results.
        Note: run_number will be assigned by the repository upon saving.
        """
        return cls(
            metadata=DiagnosticRunMetadata(
                estimator_type=estimator_type,
                estimator_version=estimator_version,
                dataset_name=dataset_name,
                mapping_direction=mapping_direction,
                num_samples=num_samples,
                scale_method=scale_method,
            ),
            accuracy=accuracy,
            reliability=reliability,
            candidates=candidates,
        )

    @classmethod
    def from_data(
        cls,
        metadata: dict[str, Any],
        accuracy: dict[str, Any],
        reliability: dict[str, Any],
        candidates: dict[str, Any],
    ) -> Self:
        """Factory for deserializing from stored data blocks."""
        return cls(
            metadata=DiagnosticRunMetadata(**metadata),
            accuracy=AccuracyLens(**accuracy),
            reliability=ReliabilityLens(**reliability),
            candidates=SpatialCandidates(**candidates),
        )
