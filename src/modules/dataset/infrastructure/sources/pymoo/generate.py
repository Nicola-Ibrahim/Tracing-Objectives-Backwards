from dataclasses import dataclass

import numpy as np

from ....domain.interfaces.base_data_source import BaseDataSource
from .algorithms import NSGAII, NSGA2Config
from .minimizer import Minimizer, MinimizerConfig
from .problems.cocoex import COCOBiObjectiveProblem


@dataclass(frozen=True)
class RawDataPayload:
    """
    Framework-agnostic container for raw data from any source.

    All data sources must provide decisions (X) and objectives (y).
    Pareto data is optional — only sources that compute it (e.g., optimization
    solvers) will populate these fields.
    """

    decisions: np.ndarray
    """Decision variable matrix, shape (n_samples, n_vars)."""

    objectives: np.ndarray
    """Objective value matrix, shape (n_samples, n_objs)."""

    pareto_set: np.ndarray | None = None
    """Pareto-optimal decisions, shape (n_pareto, n_vars). None if not computed."""

    pareto_front: np.ndarray | None = None
    """Pareto-optimal objectives, shape (n_pareto, n_objs). None if not computed."""


class PymooOptimizationGenerator(BaseDataSource):
    """
    A true Facade: The client knows NOTHING about PyMOO.
    It only passes domain-level configurations (Pydantic models, dicts, or strings).
    This class handles the creation of all PyMOO objects and runs the pipeline.
    """

    def __init__(
        self,
        problem_config: COCOBiObjectiveProblem,
        algorithm_config: NSGA2Config,
        minimizer_config: MinimizerConfig,
    ):
        """
        The client passes pure data/configs, no PyMOO objects.
        """
        self.problem_config = problem_config
        self.algorithm_config = algorithm_config
        self.minimizer_config = minimizer_config

    def generate(self) -> RawDataPayload:
        """
        The Facade hides the complexity of translating configs into PyMOO objects,
        running the minimization, and translating it back to domain entities.
        """
        # 1. Translate domain configs into PyMOO objects internally (using Factories)
        # The client doesn't know these PyMOO objects exist.
        problem = COCOBiObjectiveProblem(self.problem_config)
        algorithm = NSGAII(self.algorithm_config)

        # 2. Setup and run the Minimizer
        minimizer = Minimizer(
            problem=problem,
            algorithm=algorithm,
            config=self.minimizer_config,
        )
        opt_data = minimizer.run()

        # 3. Map PyMOO results back to domain-friendly RawDataPayload
        return RawDataPayload(
            decisions=opt_data.last_generation_data["solutions"],
            objectives=opt_data.last_generation_data["objectives"],
            pareto_set=opt_data.pareto_set,
            pareto_front=opt_data.pareto_front,
        )
