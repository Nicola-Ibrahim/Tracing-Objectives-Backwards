from pydantic import BaseModel, Field, model_validator


class AlgorithmConfig(BaseModel):
    """
    Pydantic model for NSGA-II algorithm configuration.

    Automatically validates values upon initialization.
    """

    population_size: int = Field(200, gt=0, description="Size of the population")
    crossover_prob: float = Field(
        0.9, gt=0, le=1, description="Crossover probability (0 < p ≤ 1)"
    )
    crossover_eta: float = Field(
        15.0, gt=0, description="Distribution index for crossover"
    )
    mutation_prob: float = Field(
        0.2, gt=0, le=1, description="Mutation probability (0 < p ≤ 1)"
    )
    mutation_eta: float = Field(
        20.0, gt=0, description="Distribution index for mutation"
    )

    @model_validator(mode="after")  # 👈 Required fix
    def check_probabilities(cls, values):
        cp, mp = values.crossover_prob, values.mutation_prob
        if cp is None or mp is None:
            raise ValueError("Probabilities must not be None.")
        return values
