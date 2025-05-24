from dataclasses import dataclass


@dataclass
class NSGAIIConfig:
    """Dataclass to encapsulate NSGA-II parameters with defaults"""

    population_size: int = 200
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_prob: float = 0.2
    mutation_eta: float = 20.0

    def __post_init__(self):
        # Validate and set default values for parameters
        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer.")
        if not (0 < self.crossover_prob <= 1):
            raise ValueError("Crossover probability must be in (0, 1].")
        if not (0 < self.mutation_prob <= 1):
            raise ValueError("Mutation probability must be in (0, 1].")
        if self.crossover_eta <= 0:
            raise ValueError("Crossover eta must be a positive number.")
        if self.mutation_eta <= 0:
            raise ValueError("Mutation eta must be a positive number.")
