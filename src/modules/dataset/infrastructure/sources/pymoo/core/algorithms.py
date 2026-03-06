from pydantic import BaseModel, ConfigDict, Field
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation


class NSGA2Config(BaseModel):
    """
    Pydantic model for NSGA-II algorithm configuration.
    Automatically validates values upon initialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    population_size: int = Field(..., gt=0, description="Size of the population")

    # Crossover parameters
    crossover_prob: float = Field(
        0.9, gt=0.0, le=1.0, description="Crossover probability (0 < p ≤ 1)"
    )
    crossover_eta: float = Field(
        15.0, gt=0.0, description="Distribution index for crossover"
    )

    # Mutation parameters
    mutation_prob: float = Field(
        0.2, gt=0.0, le=1.0, description="Mutation probability (0 < p ≤ 1)"
    )
    mutation_eta: float = Field(
        20.0, gt=0.0, description="Distribution index for mutation"
    )


class NSGAII(NSGA2):
    """
    Robust factory class for initializing the NSGA-II algorithm.

    Separates the Pydantic configuration from the PyMOO instantiation logic.
    """

    def __init__(self, config: NSGA2Config):
        """
        Initializes the factory with a validated NSGA2Config.
        """
        self.config = config
        super().__init__(
            pop_size=self.config.population_size,
            crossover=SBX(
                prob=self.config.crossover_prob, eta=self.config.crossover_eta
            ),
            mutation=PolynomialMutation(
                prob=self.config.mutation_prob, eta=self.config.mutation_eta
            ),
            eliminate_duplicates=True,
        )
