from functools import cached_property
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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
    history: list[dict[str, np.ndarray]] | None = Field(
        None, description="History of the algorithm's population at each generation"
    )

    class Config:
        arbitrary_types_allowed = True

    # --------- Validators for input data types ---------

    @field_validator("X", "F", "G", "CV", mode="before")
    @classmethod
    def ensure_numpy_array(cls, v):
        return np.asarray(v) if v is not None else v

    @model_validator(mode="after")
    def post_init_checks(self):
        # Handle cases where G or CV might be None for unconstrained problems
        if self.G is None:
            self.G = np.zeros((self.F.shape[0], 0))
        if self.CV is None:
            self.CV = np.zeros(self.F.shape[0])
        # Ensure CV is 1D
        if self.CV.ndim > 1:
            self.CV = self.CV.squeeze()
        return self

    # --------- Cached Properties for Core Indexing and Masks ---------

    @cached_property
    def _is_feasible_mask(self) -> np.ndarray:
        """Boolean mask indicating feasibility of each solution in the final population."""
        return self.CV <= 0.0

    @cached_property
    def _feasible_indices(self) -> np.ndarray:
        """Indices of all feasible solutions in the original array."""
        return np.flatnonzero(self._is_feasible_mask)

    @cached_property
    def _pareto_indices(self) -> np.ndarray:
        """Indices of Pareto-optimal solutions among feasible solutions."""
        if not self._is_feasible_mask.any():
            return np.array([], dtype=int)
        return NonDominatedSorting().do(
            self.F[self._is_feasible_mask], only_non_dominated_front=True
        )

    # --------- User-facing Properties ---------

    @property
    def pareto_set(self) -> np.ndarray:
        """Decision variable vectors of the Pareto-optimal solutions from the final population."""
        return self.X[self._pareto_indices]

    @property
    def pareto_front(self) -> np.ndarray:
        """Objective values of the Pareto-optimal solutions from the final population."""
        return self.F[self._pareto_indices]

    # --------- Historical Data Properties ---------

    @cached_property
    def all_historical_solutions(self) -> np.ndarray | None:
        """All decision variables from all generations, combined into a single array."""
        if not self.history:
            return None
        return np.vstack([g["X"] for g in self.history])

    @cached_property
    def all_historical_objectives(self) -> np.ndarray | None:
        """All objective values from all generations, combined into a single array."""
        if not self.history:
            return None
        return np.vstack([g["F"] for g in self.history])

    @cached_property
    def _all_historical_violations(self) -> np.ndarray | None:
        """All aggregated constraint violations from all generations, combined."""
        if not self.history:
            return None
        all_cv_list = [
            g["CV"].reshape(-1, 1)
            for g in self.history
            if g.get("CV") is not None and g["CV"].size > 0
        ]
        if all_cv_list:
            return np.vstack(all_cv_list).squeeze()

        # If no CV data, return zeros of the correct size
        return np.zeros(self.all_historical_solutions.shape[0])

    @cached_property
    def _is_feasible_historical_mask(self) -> np.ndarray | None:
        """Boolean mask indicating feasibility of all historical solutions."""
        if self._all_historical_violations is None:
            return None
        return self._all_historical_violations <= 0.0

    @cached_property
    def _pareto_indices_historical(self) -> np.ndarray | None:
        """
        Indices of Pareto-optimal solutions among all historical feasible solutions.
        The indices are relative to the combined historical arrays.
        """
        if (
            self.all_historical_objectives is None
            or self._is_feasible_historical_mask is None
        ):
            return None

        feasible_F_history = self.all_historical_objectives[
            self._is_feasible_historical_mask
        ]
        if not feasible_F_history.size:
            return np.array([], dtype=int)
        return NonDominatedSorting().do(
            feasible_F_history, only_non_dominated_front=True
        )

    @property
    def historical_pareto_set(self) -> np.ndarray | None:
        """The Pareto-optimal set across all generations (feasible solutions only)."""
        if (
            self.all_historical_solutions is None
            or self._pareto_indices_historical is None
        ):
            return None

        feasible_X_history = self.all_historical_solutions[
            self._is_feasible_historical_mask
        ]
        return feasible_X_history[self._pareto_indices_historical]

    @property
    def historical_pareto_front(self) -> np.ndarray | None:
        """The Pareto-optimal front across all generations (feasible solutions only)."""
        if (
            self.all_historical_objectives is None
            or self._pareto_indices_historical is None
        ):
            return None

        feasible_F_history = self.all_historical_objectives[
            self._is_feasible_historical_mask
        ]
        return feasible_F_history[self._pareto_indices_historical]

    # --------- High-level functionality for users ---------

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
