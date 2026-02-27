from typing import Dict, List, Tuple

from pydantic import BaseModel


class DatasetResponse(BaseModel):
    name: str
    original_objectives: List[Tuple[float, float]]
    original_decisions: List[List[float]]  # D-dimensional bounds/anchors
    bounds: Dict[str, Tuple[float, float]]  # e.g., "obj1": (min, max)
