from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class OptimizationResult:
    """
    A comprehensive container for multi-objective optimization results.

    Attributes:
        X (np.ndarray): Decision variables matrix of shape (n_samples, n_vars)
        F (np.ndarray): Objective values matrix of shape (n_samples, n_objs)
        G (np.ndarray): Constraint violations matrix of shape (n_samples, n_constraints)
        CV (np.ndarray): Combined constraint violation vector of shape (n_samples,)
        history (list, optional): Optimization history for each generation

    Properties:
        feasible_mask: Boolean mask indicating feasible solutions
        pareto_front_indices: Indices of first Pareto front solutions
        all_front_indices: Indices of solutions grouped by dominance fronts
    """

    X: np.ndarray
    F: np.ndarray
    G: np.ndarray
    CV: np.ndarray
    history: Optional[list] = None

    @property
    def feasible_mask(self) -> np.ndarray:
        """Boolean mask indicating which solutions satisfy all constraints"""
        return self.CV <= 0.0

    @property
    def pareto_front_indices(self) -> np.ndarray:
        """
        Indices of solutions in the first Pareto front (non-dominated solutions)

        Returns:
            np.ndarray: Array of indices corresponding to Pareto optimal solutions

        Note:
            Uses fast non-dominated sorting with early termination
        """
        return NonDominatedSorting().do(self.F, only_non_dominated_front=True)

    @property
    def all_front_indices(self) -> list:
        """
        Indices of solutions grouped by dominance fronts

        Returns:
            list: List of index arrays where each subarray represents a dominance front
        """
        return NonDominatedSorting().do(self.F)

    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve Pareto optimal solutions

        Returns:
            tuple: (X_pareto, F_pareto) where:
                - X_pareto: Decision variables of Pareto solutions
                - F_pareto: Objective values of Pareto solutions
        """
        idx = self.pareto_front_indices
        return self.X[idx], self.F[idx]

    def constraint_violation_summary(self) -> dict:
        """
        Generate summary statistics for constraint violations

        Returns:
            dict: Dictionary with:
                - total_violations: Count of solutions violating at least one constraint
                - violation_counts: Number of solutions violating each constraint
                - max_violations: Maximum violation per constraint
        """
        return {
            "total_violations": np.sum(~self.feasible_mask),
            "violation_counts": np.sum(self.G > 0, axis=0),
            "max_violations": np.max(self.G, axis=0),
        }
