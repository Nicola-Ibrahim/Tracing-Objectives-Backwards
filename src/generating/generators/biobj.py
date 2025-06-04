from pathlib import Path

from ...utils.archivers.base import BaseParetoArchiver
from ...utils.archivers.models import ParetoDataModel
from ...utils.archivers.npz import ParetoNPzArchiver
from ..algorithms.config import AlgorithmConfig
from ..algorithms.nsga2 import NSGAII
from ..optimizers.config import OptimizerConfig
from ..optimizers.minimize import Minimizer
from ..problems.coco.biobj import COCOBiObjectiveProblem, get_problem
from ..problems.coco.config import ProblemConfig


class ParetoDataGenerating:
    """
    Facade for generating bi-objective optimization data using COCO problems.
    This class encapsulates the configuration and execution of the optimization process,
    allowing for easy setup and execution of bi-objective optimization tasks.
    """

    def __init__(self, pareto_problem_id: int = 1):
        """
        Initialize the ParetoDataGenerating instance with default configurations.
        Args:
            pareto_problem_id (int): Identifier for the specific bi-objective problem to be used.
        """
        self.archiver: BaseParetoArchiver = ParetoNPzArchiver()
        self.coco_problem = get_problem(function_indices=pareto_problem_id)
        self.algorithm_config = None
        self.problem_config = None
        self.optimizer_config = None

        self.configure_algorithm()
        self.configure_problem()
        self.configure_optimizer()

    def configure_algorithm(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ) -> None:
        """Builder method for algorithm configuration"""
        self.algorithm_config = AlgorithmConfig(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )
        return self

    def configure_problem(self) -> None:
        """Builder method for problem configuration"""
        self.problem_config = ProblemConfig(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=self.coco_problem.lower_bounds,
            xu=self.coco_problem.upper_bounds,
        )
        return self

    def configure_optimizer(self) -> None:
        """Builder method for optimizer configuration"""
        self.optimizer_config = OptimizerConfig(
            generations=16,
            seed=42,
            save_history=False,  # Avoids deepcopy of problem object
            verbose=False,
            pf=True,
        )
        return self

    def generate(self) -> Path:
        """Execute the optimization with configured parameters"""
        if not all([self.algorithm_config, self.problem_config, self.optimizer_config]):
            raise RuntimeError(
                "All configurations must be set before running optimization"
            )

        # Configure algorithm
        algorithm = NSGAII(config=self.algorithm_config)

        # Setup problem
        problem = COCOBiObjectiveProblem(
            problem=self.coco_problem, config=self.problem_config
        )

        # Execute optimization
        optimizer = Minimizer(
            problem=problem, algorithm=algorithm, config=self.optimizer_config
        )

        result = optimizer.run()

        data = ParetoDataModel(
            pareto_set=result.pareto_set,
            pareto_front=result.pareto_front,
            problem_name=self.coco_problem.name,
            metadata={
                "algorithm": algorithm.__class__.__name__,
                "optimizer": self.optimizer_config.model_dump(),
                "problem": self.problem_config.model_dump(),
            },
        )

        return self.archiver.save(data)
