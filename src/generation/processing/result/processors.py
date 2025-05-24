from ...optimizations.result_handlers import OptimizationResultHandler
from .base import BaseResultProcessor


class OptimizationResultProcessor(BaseResultProcessor):
    """
    Processes optimization results for electric vehicle control problems.
    Maintains compatibility with the OptimizationResultHandler structure while
    providing domain-specific key naming.
    """

    def __init__(self):
        """
        Initialize with optimization results.

        Args:
            result: OptimizationResultHandler container with solution data
        """
        self.result = None

    def process(self, result: OptimizationResultHandler) -> None:
        """
        Process the optimization results to ensure they are ready for extraction.
        This method is a placeholder for any pre-processing steps if needed.
        """
        self.result = result

    def get_history(self) -> list:
        """Extract optimization history with domain-specific key naming"""
        optimization_history = []

        if self.result.history:
            for algorithm in self.result.history:
                optimization_history.append(
                    {
                        "generation": algorithm.n_gen,
                        "accel_ms2": algorithm.pop.get("X")[:, 0],
                        "decel_ms2": algorithm.pop.get("X")[:, 1],
                        "time_min": algorithm.pop.get("F")[:, 0],
                        "energy_kwh": algorithm.pop.get("F")[:, 1],
                        "feasible": algorithm.pop.get("feasible"),
                    }
                )

        return optimization_history

    def get_full_solutions(self) -> dict:
        """Get complete solution set with domain-specific naming"""
        return {
            # Decision variables
            "accel_ms2": self.result.X[:, 0],
            "decel_ms2": self.result.X[:, 1],
            # Objectives
            "time_min": self.result.F[:, 0],
            "energy_kwh": self.result.F[:, 1],
            # Feasibility
            "feasible": self.result.feasible_mask,
            # Constraint violations (direct array access)
            "constraint_violations": {
                "speed_violation": self.result.G[:, 0],
                "energy_violation": self.result.G[:, 1],
                "control_violation": self.result.G[:, 2],
            },
        }

    def get_pareto_front(self) -> dict:
        """Extract Pareto-optimal solutions with original key naming"""
        X_pareto, F_pareto = self.result.get_pareto_front()
        indices = self.result.pareto_front_indices

        return {
            # Pareto-optimal decision variables
            "accel_ms2": X_pareto[:, 0],
            "decel_ms2": X_pareto[:, 1],
            # Pareto-optimal objectives
            "time_min": F_pareto[:, 0],
            "energy_kwh": F_pareto[:, 1],
            # Feasibility status of Pareto solutions
            "feasible": self.result.feasible_mask[indices],
            # Constraint violations for Pareto front
            "constraint_violations": {
                "speed_violation": self.result.G[indices, 0],
                "energy_violation": self.result.G[indices, 1],
                "control_violation": self.result.G[indices, 2],
            },
        }

    def get_constraint_summary(self) -> dict:
        """Get constraint analysis using new OptimizationResultHandler method"""
        summary = self.result.constraint_violation_summary()
        return {
            "total_violations": summary["total_violations"],
            "speed_violations": summary["violation_counts"][0],
            "energy_violations": summary["violation_counts"][1],
            "control_violations": summary["violation_counts"][2],
            "max_speed_violation": summary["max_violations"][0],
            "max_energy_violation": summary["max_violations"][1],
            "max_control_violation": summary["max_violations"][2],
        }
