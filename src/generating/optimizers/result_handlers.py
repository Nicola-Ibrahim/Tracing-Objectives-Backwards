from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
