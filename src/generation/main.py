from dataclasses import asdict
from pathlib import Path

from .domain.value_objects import OptimizationParameters, Vehicle
from .executor import OptimizationExecutor
from .optimize.algorithms import NSGAII
from .optimize.problems.ev import EVControlProblem
from .processing.archivers.base import ResultArchiver
from .processing.archivers.pickle import PickleArchiver
from .processing.result import ResultProcessor


class OptimizationFacade:
    def __init__(self, archiver: ResultArchiver = PickleArchiver()):
        self.archiver = archiver

    def run_optimization(
        self,
        vehicle: Vehicle,
        params: OptimizationParameters,
    ) -> Path:
        # Configure algorithm
        algorithm = NSGAII()

        # Setup problem
        problem = EVControlProblem(
            target_distance_km=params.distance_km, vehicle=vehicle
        )

        # Execute optimization
        executor = OptimizationExecutor(problem, algorithm)
        result = executor.run(params.generations)

        # Process results
        processor = ResultProcessor(result)
        data = {
            "all_solutions": processor.get_full_solutions(),
            "pareto_front": processor.get_pareto_front(),
            "metadata": self._build_metadata(vehicle, params),
        }

        return self.archiver.save(data)

    def _build_metadata(self, vehicle: Vehicle, params: OptimizationParameters):
        return {
            "vehicle": asdict(vehicle),
            "parameters": asdict(params),
            "algorithm": "NSGA-II",
        }


# Usage Example
if __name__ == "__main__":
    vehicle = Vehicle(max_battery_kwh=10.0, max_speed_mps=20.0, mass_kg=1000)

    params = OptimizationParameters(
        distance_km=100, population_size=200, generations=16
    )

    facade = OptimizationFacade()
    result_file = facade.run_optimization(vehicle, params)
    print(f"Results saved to: {result_file}")
