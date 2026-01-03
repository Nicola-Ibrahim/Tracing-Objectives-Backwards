import pickle
from datetime import datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..interfaces import BaseConformalValidator, BaseOODValidator


class DecisionValidationCalibration(BaseModel):
    """Persisted bundle of fitted validators for decision validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))

    created_at: datetime = Field(default_factory=datetime.now)

    ood_validator: BaseOODValidator = Field(
        ...,
        alias="ood_calibrator",
        description="Fitted out-of-distribution validator instance.",
    )
    conformal_validator: BaseConformalValidator = Field(
        ...,
        alias="conformal_calibrator",
        description="Fitted conformal validator instance.",
    )

    # Serialize validators as pickled payloads to keep infrastructure concerns external.
    @field_serializer("ood_validator", "conformal_validator")
    def _serialize_validator(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    @field_validator("ood_validator", "conformal_validator", mode="before")
    @classmethod
    def _deserialize_validator(cls, value: Any) -> Any:
        if isinstance(value, bytes):
            return pickle.loads(value)
        return value

    @classmethod
    def from_data(
        cls,
        *,
        id: str,
        created_at: datetime,
        ood_validator: BaseOODValidator,
        conformal_validator: BaseConformalValidator,
    ) -> Self:
        return cls(
            id=id,
            created_at=created_at,
            ood_validator=ood_validator,
            conformal_validator=conformal_validator,
        )

    # ------------------------ backward-compatible props ------------------------ #
    @property
    def ood_calibrator(self) -> BaseOODValidator:
        return self.ood_validator

    @property
    def conformal_calibrator(self) -> BaseConformalValidator:
        return self.conformal_validator
