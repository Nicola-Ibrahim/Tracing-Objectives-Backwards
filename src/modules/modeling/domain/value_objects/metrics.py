from pydantic import BaseModel, Field


class Metrics(BaseModel):
    """
    Canonical shape for loss history returned by trainers.
    """

    train: list[dict[str, float]] = Field(
        default_factory=list,
        description="Performance metrics recorded for training evaluations (ordering preserved).",
    )
    test: list[dict[str, float]] = Field(
        default_factory=list,
        description="Performance metrics recorded for test evaluations (ordering preserved).",
    )
    cv: list[dict[str, float | list[float]]] = Field(
        default_factory=list,
        description="Per-fold cross-validation metrics; each entry is a metric dictionary.",
    )

    class Config:
        extra = "forbid"
