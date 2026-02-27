from pydantic import BaseModel


class DatasetResponse(BaseModel):
    name: str
    original_objectives: list[tuple[float, ...]]
    original_decisions: list[list[float]]
    bounds: dict[str, tuple[float, float]]


class DatasetDetailResponse(BaseModel):
    name: str
    X: list[list[float]]  # Decision space coordinates
    y: list[list[float]]  # Objective space coordinates
    bounds: dict[str, tuple[float, float]]  # Min/Max for each objective
