from typing import Any
import numpy as np
from pydantic import BaseModel, Field, model_validator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class HistoryEntry(BaseModel):
    """Structured data for a single generation in the optimization history."""

    X: np.ndarray
    F: np.ndarray
    G: np.ndarray | None = None
    CV: np.ndarray | None = None


class OptimizationResult(BaseModel):
    """
    Abstraction for multi-objective optimization results, including decision variables,
    objectives, constraints, and constraint violations.

    Attributes:
        X (np.ndarray): Decision variables, shape (n_samples, n_vars) of the final population
        F (np.ndarray): Objective values, shape (n_samples, n_objs) of the final population
        G (np.ndarray): Constraint violations per constraint, shape (n_samples, n_constraints)
        CV (np.ndarray): Aggregated constraint violation (e.g. max, sum), shape (n_samples,)
        history (list): Optional list of intermediate optimization results
    """

    X: np.ndarray = Field(..., description="Decision variables of the final population")
    F: np.ndarray = Field(..., description="Objective values of the final population")
    G: np.ndarray | None = Field(
        None, description="Constraint values for the final population"
    )
    CV: np.ndarray | None = Field(
        None, description="Aggregated constraint violation for the final population"
    )
    history: list[HistoryEntry] | None = Field(
        None, description="History of the algorithm's population at each generation"
    )

    @model_validator(mode="after")
    def post_init_checks(self):
        """
        Post-initialization to ensure all data is in the correct format.
        This runs after the model is initialized.
        """
        # Ensure all arrays are numpy arrays for consistency
        self.X = np.asarray(self.X)
        self.F = np.asarray(self.F)
        if self.G is not None:
            self.G = np.asarray(self.G)
        if self.CV is not None:
            self.CV = np.asarray(self.CV)

        # Handle cases where G or CV might be None for unconstrained problems
        if self.G is None:
            self.G = np.zeros((self.F.shape[0], 0))
        if self.CV is None:
            self.CV = np.zeros(self.F.shape[0])

        # Ensure CV is 1D
        if self.CV.ndim > 1:
            self.CV = self.CV.squeeze()

        # The history logic from your original code was incorrect.
        # History should be provided to the model, not created from the final population.
        return self

    
    @property
    def final_solutions(self) -> np.ndarray:
        """Decision variable vectors (X) of all feasible solutions in the final population."""
        return self.X[self._is_feasible]

    @property
    def final_objectives(self) -> np.ndarray:
        """Objective values (F) for all feasible solutions in the final population."""
        return self.F[self._is_feasible]

  
    @property
    def pareto_set(self) -> np.ndarray:
        """
        Decision variable vectors of Pareto-optimal solutions (feasible only)

        Returns:
            np.ndarray: Decision variables of Pareto-optimal solutions
        """
        return self.final_solutions[self._get_final_pareto_front_indices()]

    @property
    def pareto_front(self) -> np.ndarray:
        """
        Objective values of Pareto-optimal solutions (feasible only)
        Returns:
            np.ndarray: Objective values of Pareto-optimal solutions
        """
        return self.final_objectives[self._get_final_pareto_front_indices()]

    @property
    def non_pareto_objectives(self) -> np.ndarray:
        """Objective values of feasible but non-Pareto solutions"""
        mask = np.ones(len(self.final_objectives), dtype=bool)
        mask[self._get_final_pareto_front_indices()] = False
        return self.final_objectives[mask]


    @property
    def pareto_solution_indices(self) -> np.ndarray:
        """Indices of Pareto-optimal solutions in the original array"""
        return self._get_final_feasible_indices[self._get_final_pareto_front_indices]

    # =================================================================
    # Public Methods: High-level functionality for users
    # =================================================================

    def get_constraint_violation_stats(self) -> dict[str, Any]:
        """
        Returns a dictionary of summary statistics for constraint violations
        in the final population.
        """
        if self.G is None or self.G.size == 0:
            return {
                "total_violations": 0,
                "violation_counts": np.array([], dtype=int),
                "max_violations": np.array([], dtype=float),
            }
        
        return {
            "total_violations": np.sum(~self.is_feasible),
            "violation_counts": np.sum(self.G > 0, axis=0),
            "max_violations": np.max(self.G, axis=0),
        }

    def get_history_data(self) -> list[dict[str, np.ndarray]]:
        """
        Extracts all optimization history data into a structured format.

        Returns:
            list of dictionaries with keys: 'X', 'F', 'G', 'CV' for each generation.
        """
        if not self.history:
            return None

        history_data = []
        for entry in self.history:
            history_data.append(
                {
                    "X": entry.X,
                    "F": entry.F,
                    "G": entry.G,
                    "CV": entry.CV,
                }
            )
        return history_data

    

    # =================================================================
    # Private Helper Properties: For internal use only
    # =================================================================

    def _is_feasible(self) -> np.ndarray:
        """Boolean mask indicating feasibility of each solution in the final population."""
        return self.CV <= 0.0


    def _get_final_pareto_front_indices(self) -> np.ndarray:
        """(Private) Indices (relative to the final feasible set) of Pareto-optimal solutions."""
        if not self._is_feasible.any():
            return np.array([], dtype=int)
        return NonDominatedSorting().do(self.final_objectives, only_non_dominated_front=True)
    
    def _get_final_all_fronts_indices(self) -> list[np.ndarray]:
        """(Private) Full non-dominated sorting of the final feasible solutions into fronts."""
        if not self._is_feasible.any():
            return []
        return NonDominatedSorting().do(self.final_objectives, only_non_dominated_front=False)

    def _get_final_feasible_indices(self) -> np.ndarray:
        """(Private) Indices of all feasible solutions in the original final population array."""
        return np.flatnonzero(self._is_feasible)

    def _get_final_pareto_solution_indices(self) -> np.ndarray:
        """(Private) Indices of Pareto-optimal solutions in the original final population array."""
        return self._get_final_feasible_indices[self._get_final_pareto_front_indices]



class OptimizationResultProcessor:
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
        indices = self.result _get_final_pareto_front_indices()]

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
