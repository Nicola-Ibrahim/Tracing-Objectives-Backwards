from dataclasses import dataclass


@dataclass
class MinimizerConfig:
    """Dataclass to encapsulate optimization parameters with defaults"""

    generations: int = 16  # Number of generations for the optimization
    seed: int = 42  # Random seed for reproducibility
    verbose: bool = False  # Verbosity flag for logging
    save_history: bool = True  # Flag to save optimization history
    pf: bool = True  # Flag to save Pareto front
