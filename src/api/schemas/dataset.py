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
    is_pareto: list[bool]  # Mask for Pareto-optimal points
    bounds: dict[str, tuple[float, float]]  # Min/Max for each objective


class DatasetGenerationRequest(BaseModel):
    function_id: int
    population_size: int = 200
    n_var: int = 2
    generations: int = 20
    dataset_name: str
