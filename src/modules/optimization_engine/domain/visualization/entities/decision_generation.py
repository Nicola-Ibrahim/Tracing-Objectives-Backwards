from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class GeneratorRunResult:
    """Container for a single inverse model generation run."""

    name: str
    decisions: npt.NDArray[np.float64]
    predicted_objectives: npt.NDArray[np.float64]


@dataclass
class DecisionGenerationVisualizationData:
    """
    Structured payload for decision-generation comparisons.

    Attributes:
        pareto_front: Raw pareto front values in objective space.
        target_objective: Raw target objective as a 1D array.
        generators: Ordered list of generator results to visualize.
    """

    pareto_front: npt.NDArray[np.float64]
    target_objective: npt.NDArray[np.float64]
    generators: list[GeneratorRunResult]

    def __post_init__(self) -> None:
        self.pareto_front = np.asarray(self.pareto_front, dtype=float)
        self.target_objective = np.asarray(self.target_objective, dtype=float).reshape(
            -1
        )
        self.generators = list(self.generators)
