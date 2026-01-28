from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel

from ..entities.accuracy_lens import AccuracyLens
from ..entities.reliability_lens import ReliabilityLens
from ..value_objects.diagnostic_run_metadata import DiagnosticRunMetadata
from ..value_objects.estimator import Estimator


class DiagnosticResult(BaseModel):
    """
    Aggregate Root for Inverse Model Diagnostics.
    Enforces the Dual-Lens framework with serialization support.
    """

    metadata: DiagnosticRunMetadata
    accuracy: AccuracyLens
    reliability: ReliabilityLens

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}
        json_decoders = {datetime: datetime.fromisoformat}

    @classmethod
    def create(
        cls,
        estimator: Estimator,
        dataset_name: str,
        num_samples: int,
        scale_method: str,
        accuracy: AccuracyLens,
        reliability: ReliabilityLens,
    ) -> Self:
        """
        Factory for new diagnostic results.
        Note: run_number will be assigned by the repository upon saving.
        """
        return cls(
            metadata=DiagnosticRunMetadata(
                estimator=estimator,
                dataset_name=dataset_name,
                num_samples=num_samples,
                scale_method=scale_method,
            ),
            accuracy=accuracy,
            reliability=reliability,
        )

    @classmethod
    def from_data(
        cls,
        metadata: dict[str, Any],
        accuracy: dict[str, Any],
        reliability: dict[str, Any],
    ) -> Self:
        """Factory for deserializing from stored data blocks."""
        return cls(
            metadata=DiagnosticRunMetadata(**metadata),
            accuracy=AccuracyLens(**accuracy),
            reliability=ReliabilityLens(**reliability),
        )
