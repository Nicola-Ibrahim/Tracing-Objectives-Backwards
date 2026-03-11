from pydantic import BaseModel

from ..value_objects.empirical_distribution import EmpiricalDistribution
from ..value_objects.reliability_summary import ReliabilitySummary


class ReliabilityLens(BaseModel):
    """
    Reliability assessment of the inverse engine.
    Focuses on calibration (PIT), sharpness (CRPS), and diversity.
    """

    calibration_error: float  # MACE
    crps: float
    pit_profile: EmpiricalDistribution
    calibration_curve: EmpiricalDistribution
    summary: ReliabilitySummary
