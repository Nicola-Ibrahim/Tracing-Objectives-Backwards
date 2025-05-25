from dataclasses import asdict
from pathlib import Path

from ..processing.archivers.base import BaseResultArchiver
from ..processing.result.processors import BaseResultProcessor
from .algorithms import NSGAII
from .algorithms.nsga2 import NSGAIIConfig
from .optimizers.minimizers import Minimizer
from .optimizers.optim_config import MinimizerConfig
from .problems.ev.electric_vehicle import EVControlProblem
from .problems.ev.specs import ProblemSpec
from .problems.ev.vehicle import Vehicle


class OptimizationFacade:
    def __init__(self, archiver: BaseResultArchiver, processor: BaseResultProcessor):
        self.archiver = archiver
        self.processor = processor

    def run_optimization(
        self,
        algorithm_config: NSGAIIConfig,
        problem_spec: ProblemSpec,
        vehicle: Vehicle,
        opt_config: MinimizerConfig,
    ) -> Path:
        # Configure algorithm
        algorithm = NSGAII(config=algorithm_config)

        # Setup problem
        problem = EVControlProblem(spec=problem_spec, vehicle=Vehicle)

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

    def _build_metadata(self, vehicle: Vehicle, opt_config: MinimizerConfig):
        return {
            "config": {
                "vehicle": asdict(vehicle),
                "optimization": asdict(opt_config),
            },
            "algorithm": self.algorithm.__class__.__name__,
        }
