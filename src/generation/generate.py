from pathlib import Path

from .optimizations.algorithms.nsga2 import NSGAIIConfig
from .optimizations.facade import OptimizationFacade
from .optimizations.optimizers.optim_config import MinimizerConfig
from .optimizations.problems.specs import ProblemSpec
from .optimizations.problems.vehicle import Vehicle
from .processing.archivers.pickle import PickleResultArchiver
from .processing.result.processors import OptimizationResultProcessor

if __name__ == "__main__":
    problem_spec = ProblemSpec(
        target_distance_km=150,
        vehicle=Vehicle(
            max_battery_kwh=24,
            max_speed_mps=25,
            mass_kg=1500,
            min_acceleration=0.2,
            max_acceleration=2.0,
        ),
    )

    algorithm_config = NSGAIIConfig(
        population_size=200,
        crossover_prob=0.9,
        crossover_eta=15.0,
        mutation_prob=0.2,
        mutation_eta=20.0,
    )

    opt_config = (
        MinimizerConfig(
            generations=16, seed=42, verbose=False, save_history=True, pf=True
        ),
    )

    archiver = PickleResultArchiver(output_dir=Path("data/raw"))
    processor = OptimizationResultProcessor()
    facade = OptimizationFacade(archiver=archiver, processor=processor)

    result_file = facade.run_optimization(
        algorithm_config=algorithm_config,
        problem_spec=problem_spec,
        opt_config=opt_config,
    )
    print(f"Results saved to: {result_file}")
