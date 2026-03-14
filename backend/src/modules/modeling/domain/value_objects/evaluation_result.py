from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    train: list[dict[str, float]] = Field(default_factory=list)
    test: list[dict[str, float]] = Field(default_factory=list)
    cv: list[dict[str, float]] = Field(default_factory=list)
