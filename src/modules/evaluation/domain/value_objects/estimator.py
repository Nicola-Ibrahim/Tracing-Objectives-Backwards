from pydantic import BaseModel


class Estimator(BaseModel):
    type: str
    version: int
    mapping_direction: str
