from typing import Literal

from pydantic import BaseModel, Field


class EngineCandidate(BaseModel):
    """
    Identifies a specific inverse engine and version for evaluation.
    This is an application-layer concern (part of the command).
    """

    solver_type: str = Field(..., examples=["GBPI"])
    version: int | None = Field(
        default=None,
        description="Specific integer version number. If None, latest is used.",
    )


class RunDiagnosticsCommand(BaseModel):
    """
    Command for the full evaluation suite.
    Infrastructure concerns (task_id) are removed.
    """

    dataset_name: str = Field(..., examples=["cocoex_f5"])

    inverse_engine_candidates: list[EngineCandidate] = Field(
        ...,
        description="List of engine candidates to compare.",
    )

    num_samples: int = Field(default=200, description="K candidates per target")
    random_state: int = 42

    scale_method: Literal["sd", "mad", "iqr"] = Field(
        default="sd", description="sd | mad | iqr"
    )
