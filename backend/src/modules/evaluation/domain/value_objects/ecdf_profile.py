from pydantic import BaseModel


class ECDFProfile(BaseModel):
    """
    Empirical Cumulative Distribution Function profile.
    Used in objective space for all engines (error distribution),
    and in decision space for interval engines (coverage curve).
    """

    x_values: list[float]  # sorted values
    cumulative_probabilities: list[float]  # corresponding cumulative probabilities
