import pickle
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..interfaces import BaseConformalCalibrator, BaseOODCalibrator


class DecisionValidationCalibration(BaseModel):
    """Persisted bundle of fitted calibrators for decision validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))

    created_at: datetime = Field(default_factory=datetime.now)

    ood_calibrator: BaseOODCalibrator = Field(
        ..., description="Fitted out-of-distribution calibrator instance."
    )
    conformal_calibrator: BaseConformalCalibrator = Field(
        ..., description="Fitted conformal calibrator instance."
    )

    # Serialize calibrators as pickled payloads to keep infrastructure concerns external.
    @field_serializer("ood_calibrator", "conformal_calibrator")
    def _serialize_calibrator(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    @field_validator("ood_calibrator", "conformal_calibrator", mode="before")
    @classmethod
    def _deserialize_calibrator(cls, value: Any) -> Any:
        if isinstance(value, bytes):
            return pickle.loads(value)
        return value

    @classmethod
    def from_data(
        cls,
        *,
        id: str,
        created_at: datetime,
        ood_calibrator: BaseOODCalibrator,
        conformal_calibrator: BaseConformalCalibrator,
    ) -> Self:
        return cls(
            id=id,
            created_at=created_at,
            ood_calibrator=ood_calibrator,
            conformal_calibrator=conformal_calibrator,
        )
