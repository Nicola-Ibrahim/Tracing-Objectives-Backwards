import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DataSplit(BaseModel):
    """
    Encapsulates train/test splitting metadata and index arrays.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    train_indices: np.ndarray = Field(default_factory=lambda: np.array([], dtype=int))
    test_indices: np.ndarray = Field(default_factory=lambda: np.array([], dtype=int))
    split_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_state: int = Field(default=42)

    @property
    def has_test_split(self) -> bool:
        return len(self.test_indices) > 0
