from ..domain.entities import ParetoFront
from ..domain.value_objects import OptimizationResult


class ResultProcessor:
    def __init__(self, result: OptimizationResult):
        self.result = result

    def get_full_solutions(self):
        return {
            "accel_ms2": self.result.X[:, 0],
            "decel_ms2": self.result.X[:, 1],
            "time_min": self.result.F[:, 0],
            "energy_kwh": self.result.F[:, 1],
            "feasible": self.result.feasible_mask,
        }

    def get_pareto_front(self):
        pareto = ParetoFront(self.result)
        indices = pareto.get_indices()
        return {
            "accel_ms2": self.result.X[indices, 0],
            "decel_ms2": self.result.X[indices, 1],
            "time_min": self.result.F[indices, 0],
            "energy_kwh": self.result.F[indices, 1],
            "feasible": self.result.feasible_mask[indices],
        }
