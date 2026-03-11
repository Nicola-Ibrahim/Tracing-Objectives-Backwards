from pydantic import BaseModel

from ..value_objects.accuracy_summary import AccuracySummary
from ..value_objects.empirical_distribution import EmpiricalDistribution


class AccuracyLens(BaseModel):
    """
    Accuracy assessment of the inverse engine.
    Focuses on discrepancy between generated designs and targets in objective space.
    """

    discrepancy_profile: EmpiricalDistribution
    summary: AccuracySummary
