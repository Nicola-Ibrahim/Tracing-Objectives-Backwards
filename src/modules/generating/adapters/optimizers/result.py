from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class OptimizationResult:
    """
    Abstraction for multi-objective optimization results, including decision variables,
    objectives, constraints, and constraint violations.

    Attributes:
        X (np.ndarray): Decision variables, shape (n_samples, n_vars)
        F (np.ndarray): Objective values, shape (n_samples, n_objs)
        G (np.ndarray): Constraint violations per constraint, shape (n_samples, n_constraints)
        CV (np.ndarray): Aggregated constraint violation (e.g. max, sum), shape (n_samples,)
        history (Optional[List]): Optional list of intermediate optimization results
    """

    X: np.ndarray
    F: np.ndarray
    G: np.ndarray
    CV: np.ndarray
    history: Optional[List] = None

    def __post_init__(self):
        """Ensure CV is 1D for proper boolean indexing"""
        if self.CV.ndim > 1:
            self.CV = self.CV.squeeze()

    @property
    def is_feasible(self) -> np.ndarray:
        """Boolean mask indicating feasibility of each solution."""
        return self.CV <= 0.0

    @property
    def solutions(self) -> np.ndarray:
        """Feasible decision variable vectors (X)"""
        return self.X[self.is_feasible]

    @property
    def objectives(self) -> np.ndarray:
        """Objective values for all feasible solutions (F)"""
        return self.F[self.is_feasible]

    @property
    def pareto_front_indices(self) -> np.ndarray:
        """Indices (within feasible set) of Pareto-optimal solutions"""
        if not self.is_feasible.any():
            return np.array([], dtype=int)
        return NonDominatedSorting().do(self.objectives, only_non_dominated_front=True)

    @property
    def pareto_set(self) -> np.ndarray:
        """
        Decision variable vectors of Pareto-optimal solutions (feasible only)

        Returns:
            np.ndarray: Decision variables of Pareto-optimal solutions
        """
        return self.solutions[self.pareto_front_indices]

    @property
    def pareto_front(self) -> np.ndarray:
        """
        Objective values of Pareto-optimal solutions (feasible only)
        Returns:
            np.ndarray: Objective values of Pareto-optimal solutions
        """
        return self.objectives[self.pareto_front_indices]

    @property
    def non_pareto_objectives(self) -> np.ndarray:
        """Objective values of feasible but non-Pareto solutions"""
        mask = np.ones(len(self.objectives), dtype=bool)
        mask[self.pareto_front_indices] = False
        return self.objectives[mask]

    @property
    def solution_indices(self) -> np.ndarray:
        """Indices of all feasible solutions in the original array"""
        return np.flatnonzero(self.is_feasible)

    @property
    def pareto_solution_indices(self) -> np.ndarray:
        """Indices of Pareto-optimal solutions in the original array"""
        return self.solution_indices[self.pareto_front_indices]

    @property
    def constraint_violation_stats(self) -> Dict[str, np.ndarray]:
        """
        Summary statistics for constraint violations.

        Returns:
            dict: {
                'total_violations': int,
                'violation_counts': np.ndarray,
                'max_violations': np.ndarray
            }
        """
        return {
            "total_violations": np.sum(~self.is_feasible),
            "violation_counts": np.sum(self.G > 0, axis=0),
            "max_violations": np.max(self.G, axis=0),
        }

    @property
    def all_fronts_indices(self) -> List[np.ndarray]:
        """
        Full non-dominated sorting of feasible solutions into fronts

        Returns:
            List of arrays, each representing one front's indices (relative to feasible only)
        """
        if not self.is_feasible.any():
            return []
        return NonDominatedSorting().do(self.objectives)


class OptimizationResultProcessor(BaseResultProcessor):
    """
    Processes optimization results for electric vehicle control problems.
    Maintains compatibility with the OptimizationResult structure while
    providing domain-specific key naming.
    """

    def __init__(self):
        """
        Initialize with optimization results.

        Args:
            result: OptimizationResult container with solution data
        """
        self.result = None

    def process(self, result: OptimizationResult) -> None:
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
        """Get constraint analysis using new OptimizationResult method"""
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
