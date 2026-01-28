from pydantic import BaseModel


class AccuracySummary(BaseModel):
    mean_best_shot: float
    median_best_shot: float
    mean_bias: float
    mean_dispersion: float
