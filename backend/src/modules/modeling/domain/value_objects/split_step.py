from typing import Literal

from pydantic import BaseModel


class SplitConfig(BaseModel):
    strategy: Literal["holdout", "k-fold", "stratified"] = "holdout"
    test_size: float = 0.2
    random_state: int = 42


class SplitStep(BaseModel):
    config: SplitConfig

    class Config:
        arbitrary_types_allowed = True
