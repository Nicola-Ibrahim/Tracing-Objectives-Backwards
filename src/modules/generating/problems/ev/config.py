from pydantic import BaseModel, Field


class ProblemConfig(BaseModel):
    """
    Problem specification for optimization algorithms.
    Contains the target distance for the mission and the vehicle configuration.
    """

    target_distance_km: float = Field(
        ..., ge=0.0, description="Mission distance in kilometers"
    )

    n_var: int = Field(
        ...,
        ge=1,
        description="Number of decision variables in the optimization problem",
    )
    n_obj: int = Field(..., ge=1, description="Number of objectives to optimize")
    n_constr: int = Field(
        0, ge=0, description="Number of constraints in the optimization problem"
    )
    xl: float = Field(..., description="Lower bounds for decision variables")
    xu: float = Field(..., description="Upper bounds for decision variables")
