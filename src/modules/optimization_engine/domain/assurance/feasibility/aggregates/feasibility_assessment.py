import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...shared.reasons import FeasibilityFailureReason
from ..value_objects.score import Score
from ..value_objects.suggestions import Suggestions


class FeasibilityAssessment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: np.typing.NDArray
    target_normalized: np.typing.NDArray
    is_feasible: bool
    score: Score | None = None
    reason: FeasibilityFailureReason | None = None
    suggestions: Suggestions | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reason(self):
        if self.is_feasible and self.reason is not None:
            raise ValueError("Feasible assessments cannot include a failure reason.")
        return self

    @property
    def suggestion_count(self) -> int:
        return self.suggestions.count if self.suggestions is not None else 0
