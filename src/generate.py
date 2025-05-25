from pathlib import Path

from generation.optimizations.algorithms.nsga2 import NSGAIIConfig
from generation.optimizations.facade import OptimizationFacade
from generation.optimizations.optimizers.optim_config import MinimizerConfig
from generation.optimizations.problems.ev.specs import ProblemSpec
from generation.optimizations.problems.ev.vehicle import Vehicle
from generation.processing.archivers.pickle import PickleResultArchiver
from generation.processing.result.processors import OptimizationResultProcessor

if __name__ == "__main__":
    # Optimization targets
    target_distances_km = [5, 10, 15, 20]

    # Run optimizations
    for distance in target_distances_km:
        print(f"⚡ Optimizing {distance}km scenario...")

        vehicle = Vehicle()
        problem_spec = ProblemSpec(
            target_distance_km=distance,
            n_var=2,
            n_obj=2,
            n_constr=3,
            xl=vehicle.min_acceleration,
            xu=vehicle.max_acceleration,
        )

        algorithm_config = NSGAIIConfig(
            population_size=200,
            crossover_prob=0.9,
            crossover_eta=15.0,
            mutation_prob=0.2,
            mutation_eta=20.0,
        )

        opt_config = MinimizerConfig(
            generations=16,
            seed=42,
            verbose=False,
            save_history=True,
            pf=True,
        )

        archiver = PickleResultArchiver(output_dir=Path("data/raw"))
        processor = OptimizationResultProcessor()
        facade = OptimizationFacade(archiver=archiver, processor=processor)

        result_file = facade.run_optimization(
            algorithm_config=algorithm_config,
            problem_spec=problem_spec,
            vehicle=vehicle,
            opt_config=opt_config,
        )
        print(f"Results saved to: {result_file}")
