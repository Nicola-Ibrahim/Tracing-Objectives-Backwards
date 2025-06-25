from dataclasses import asdict
from pathlib import Path

from ....shared.infrastructure.archivers.base import BaseParetoDataRepository
from ....shared.infrastructure.archivers.pickle import PickleParetoDataRepository
from ..algorithms.config import NSGA2Config
from ..algorithms.nsga2 import NSGAII
from ..optimizers.config import OptimizerConfig
from ..optimizers.minimize import Minimizer
from ..problems.ev.config import ProblemConfig
from ..problems.ev.electric_vehicle import EVControlProblem
from ..problems.ev.vehicle import Vehicle
from ..result.processors import BaseResultProcessor, OptimizationResultProcessor


class ElectricalVechicleParetoGenerating:
    """
    Facade for generating Pareto optimization data for electric vehicle control problems.
    This class encapsulates the configuration and execution of the optimization process,
    allowing for easy setup and execution of multi-objective optimization tasks.
    It uses NSGA-II algorithm and processes results using a specified archiver and processor.
    """

    def __init__(
        self, archiver: BaseParetoDataRepository, processor: BaseResultProcessor
    ):
        self.archiver = archiver
        self.processor = processor

    def generate(
        self,
        algorithm_config: NSGA2Config,
        problem_spec: ProblemConfig,
        vehicle: Vehicle,
        opt_config: OptimizerConfig,
    ) -> Path:
        # Configure algorithm
        algorithm = NSGAII(config=algorithm_config)

        # Setup problem
        problem = EVControlProblem(spec=problem_spec, vehicle=vehicle)

        # Execute optimization
        optimizer = Minimizer(problem=problem, algorithm=algorithm, config=opt_config)

        result = optimizer.run()

        # Process results
        processor = self.processor.process(result)
        data = {
            "all_solutions": processor.get_full_solutions(),
            "pareto_front": processor.get_pareto_front(),
            "metadata": self._build_metadata(problem_spec.vehicle, opt_config),
        }

        return self.archiver.save(data)

    def _build_metadata(self, vehicle: Vehicle, opt_config: OptimizerConfig):
        return {
            "config": {
                "vehicle": asdict(vehicle),
                "optimization": asdict(opt_config),
            },
            "algorithm": self.algorithm.__class__.__name__,
        }


def run_optimization():
    problem_spec = ProblemConfig(
        target_distance_km=distance,
        n_var=2,
        n_obj=2,
        n_constr=3,
        xl=vehicle.min_acceleration,
        xu=vehicle.max_acceleration,
    )

    algorithm_config = NSGA2Config(
        population_size=200,
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.2,
        mutation_eta=20.0,
    )

    opt_config = OptimizerConfig(
        generations=16,
        seed=42,
        verbose=False,
        save_history=True,
        pf=True,
    )

    archiver = PickleParetoDataRepository(output_dir=Path("data/raw"))
    processor = OptimizationResultProcessor()

    result_file = facade.run_optimization(
        algorithm_config=algorithm_config,
        problem_spec=problem_spec,
        vehicle=vehicle,
        opt_config=opt_config,
    )
    print(f"Results saved to: {result_file}")
