from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

from .base import BaseOptimizationAlgorithm
from .configs import NSGAIIConfig


class NSGAII(BaseOptimizationAlgorithm):
    """Wrapper class for NSGA-II algorithm with proper initialization"""

    def __new__(self, config: NSGAIIConfig) -> NSGA2:
        return NSGA2(
            pop_size=config.population_size,
            crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
            mutation=PolynomialMutation(
                prob=config.mutation_prob, eta=config.mutation_eta
            ),
            eliminate_duplicates=True,
        )
