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

        # Get objective values for all feasible solutions
        feasible_F = self.F[self._feasible_indices]

        # Get the non-dominated indices within the feasible subset
        non_dominated_indices_in_feasible_set = NonDominatedSorting().do(
            feasible_F, only_non_dominated_front=True
        )

        # Map these indices back to the original full array
        return self._feasible_indices[non_dominated_indices_in_feasible_set]

    # --------- User-facing Properties ---------

    @property
    def pareto_set(self) -> np.ndarray:
        """Decision variable vectors of the Pareto-optimal solutions from the final population."""
        return self.X[self._pareto_indices]

    @property
    def pareto_front(self) -> np.ndarray:
        """Objective values of the Pareto-optimal solutions from the final population."""
        return self.F[self._pareto_indices]

    # --------- Historical Data Properties (now includes non-dominated sorting) ---------

    @cached_property
    def historical_pareto_data(self) -> dict[str, Any] | None:
        """
        Processes the entire history and returns the combined Pareto-optimal data.
        Returns None if history is empty.
        """
        if not self.history:
            return None

        all_solutions = []
        all_objectives = []
        all_violations = []

        # Step 1: Collect ALL solutions, objectives, and violations from EVERY generation.
        for generation_data in self.history:
            # Correctly access the data using the keys from the dictionary
            all_solutions.append(generation_data.get("X"))
            all_objectives.append(generation_data.get("F"))

            cv_data = generation_data.get("CV")
            if cv_data is None:
                # If CV is None, create a zero array of the correct size
                cv_data = np.zeros(generation_data.get("X").shape[0])
            all_violations.append(cv_data)

        # Step 2: Combine all collected data into single, large NumPy arrays.
        combined_solutions = np.vstack(all_solutions)
        combined_objectives = np.vstack(all_objectives)

        # We use vstack for all arrays to ensure they have the same number of rows.
        combined_violations = np.vstack(all_violations).squeeze()

        # Step 3: Create a single feasibility mask for the entire historical population.
        is_feasible_mask = combined_violations <= 0.0

        # Step 4: Filter the solutions and objectives using the mask.
        feasible_solutions = combined_solutions[is_feasible_mask]
        feasible_objectives = combined_objectives[is_feasible_mask]

        if feasible_solutions.size == 0:
            return None

        # Step 5: Perform one final non-dominated sort on the entire set of feasible solutions.
        nd_sorting = NonDominatedSorting()
        non_dominated_indices = nd_sorting.do(
            feasible_objectives, only_non_dominated_front=False
        )
        non_dominated_indices = np.hstack(non_dominated_indices)

        return {
            "solutions": feasible_solutions[non_dominated_indices],
            "objectives": feasible_objectives[non_dominated_indices],
        }

    @property
    def historical_solutions(self) -> np.ndarray | None:
        """The Pareto-optimal set across all generations (feasible solutions only)."""
        if self.historical_pareto_data:
            return self.historical_pareto_data["solutions"]
        return None

    @property
    def historical_objectives(self) -> np.ndarray | None:
        """The Pareto-optimal front across all generations (feasible solutions only)."""
        if self.historical_pareto_data:
            return self.historical_pareto_data["objectives"]
        return None

    # The rest of the methods and properties can be removed from the previous implementation
    # as they are replaced by the more robust historical_solutions/front properties.
    # We will keep the get_constraint_violation_stats method as it's useful.
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
            "total_violations": np.sum(self.CV > 0),
            "violation_counts": np.sum(self.G > 0, axis=0),
            "max_violations": np.max(self.G, axis=0),
        }
