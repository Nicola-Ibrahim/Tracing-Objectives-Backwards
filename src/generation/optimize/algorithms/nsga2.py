from dataclasses import dataclass

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

from .base import OptimizationAlgorithm


@dataclass
class NSGAIIConfig:
    """Dataclass to encapsulate NSGA-II parameters with defaults"""

    population_size: int = 200
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_prob: float = 0.2
    mutation_eta: float = 20.0


class NSGAII(OptimizationAlgorithm):
    def __init__(self, **kwargs):
        # Use dataclass for configuration with type-safe defaults
        self.config = NSGAIIConfig(**kwargs)

        # Validate parameters
        if self.config.population_size <= 0:
            raise ValueError("Population size must be positive")
        if not (0 <= self.config.crossover_prob <= 1):
            raise ValueError("Crossover probability must be between 0 and 1")

    def configure(self) -> NSGA2:
        """Create configured NSGA-II algorithm instance"""
        return NSGA2(
            pop_size=self.config.population_size,
            crossover=SBX(
                prob=self.config.crossover_prob, eta=self.config.crossover_eta
            ),
            mutation=PolynomialMutation(
                prob=self.config.mutation_prob, eta=self.config.mutation_eta
            ),
            eliminate_duplicates=True,
        )
