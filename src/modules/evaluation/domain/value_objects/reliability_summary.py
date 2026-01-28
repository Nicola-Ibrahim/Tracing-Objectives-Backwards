from pydantic import BaseModel


class ReliabilitySummary(BaseModel):
    mean_crps: float
    mean_diversity: float
    mean_interval_width: float
