"""Optional entity capturing findings from individual feasibility validators."""

from pydantic import BaseModel, ConfigDict

from ...shared.reasons import FeasibilityFailureReason


class AssessmentFinding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_feasible: bool
    reason: FeasibilityFailureReason | None = None
    score: float | None = None
    message: str | None = None


__all__ = ["AssessmentFinding"]
