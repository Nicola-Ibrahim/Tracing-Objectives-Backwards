from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vehicle:
    max_battery_kwh: float
    max_speed_mps: float
    mass_kg: float

    def __post_init__(self):
        if any(
            v <= 0 for v in [self.max_battery_kwh, self.max_speed_mps, self.mass_kg]
        ):
            raise ValueError("All vehicle parameters must be positive")


@dataclass
class OptimizationParameters:
    distance_km: float
    population_size: int
    generations: int


@dataclass
class OptimizationResult:
    X: np.ndarray  # Decision variables
    F: np.ndarray  # Objective values
    G: np.ndarray  # Constraint violations
    CV: np.ndarray  # Combined constraint violations

    @property
    def feasible_mask(self):
        return self.CV <= 0.0
