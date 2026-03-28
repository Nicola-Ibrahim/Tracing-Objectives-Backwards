from pydantic import BaseModel

from .calibration_curve import CalibrationCurve
from .ecdf_profile import ECDFProfile
from .pit_profile import PITProfile


class DecisionSpaceDistributionAssessment(BaseModel):
    """
    Probabilistic assessment for full P(y|x) engines: MDN, CAVA.
    Calibration via PIT uniformity.

    mace:               Mean Absolute Calibration Error — deviation from uniform PIT.
    mean_crps:          Continuous Ranked Probability Score — sharpness/calibration.
    mean_interval_width: average width of the predictive intervals.
    mean_diversity:     average pairwise distance between generated samples.
    """

    pit_profile: PITProfile
    calibration_curve: CalibrationCurve
    mace: float
    mean_crps: float
    mean_interval_width: float
    mean_diversity: float


class DecisionSpaceIntervalAssessment(BaseModel):
    """
    Probabilistic assessment for prediction interval engines: GBPI.
    Calibration via empirical coverage (ECDF).

    nominal_coverage:   requested coverage levels e.g. [0.5, 0.8, 0.9, 0.95].
    empirical_coverage: actually observed coverage at each nominal level.
    mean_coverage_error: mean absolute deviation between nominal and empirical coverage.
    mean_interval_width: average width of the prediction intervals.
    mean_winkler_score: penalises interval width and coverage violations jointly.
    """

    ecdf_profile: ECDFProfile
    mean_coverage_error: float
    mean_interval_width: float
    mean_winkler_score: float
