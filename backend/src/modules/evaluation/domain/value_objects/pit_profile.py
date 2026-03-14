from pydantic import BaseModel


class PITProfile(BaseModel):
    """
    Probability Integral Transform histogram.
    Uniform → perfectly calibrated.
    Reused for both discrepancy profiling (objective space)
    and PIT calibration (distribution assessment).
    """

    bin_edges: list[float]
    counts: list[int]
