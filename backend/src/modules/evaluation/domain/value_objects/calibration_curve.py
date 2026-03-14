from pydantic import BaseModel


class CalibrationCurve(BaseModel):
    """
    Predicted vs empirical coverage at each quantile level.
    Diagonal → perfectly calibrated.
    Exclusive to full distribution engines: MDN, CVAE.
    """

    nominal_coverage: list[float]
    empirical_coverage: list[float]
