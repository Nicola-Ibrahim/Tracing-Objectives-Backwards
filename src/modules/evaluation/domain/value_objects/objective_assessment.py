from pydantic import BaseModel

from .ecdf_profile import ECDFProfile


class ObjectiveSpaceAssessment(BaseModel):
    """
    Point accuracy in y-space. Applies to all engines regardless of capability.

    ecdf_profile: empirical distribution of objective-space errors.
    mean_best_shot:      mean closest generated design to each target.
    median_best_shot:    median closest generated design to each target.
    mean_bias:           systematic offset between predictions and targets.
    mean_dispersion:     spread of generated designs around the target.
    """

    ecdf_profile: ECDFProfile
    mean_best_shot: float
    median_best_shot: float
    mean_bias: float
    mean_dispersion: float
