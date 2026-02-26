from functools import cached_property
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MinimizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    generations: int = Field(
        ..., gt=0, description="Number of generations must be > 0", example=100
    )
    save_history: bool = Field(
        ..., description="Flag to save optimization history", example=True
    )
    verbose: bool = Field(..., description="Flag for verbose output", example=False)

    seed: int = Field(
        42, ge=0, description="Random seed must be non-negative", example=42
    )
    # Note: In pymoo, `pf` usually expects a numpy array of the true Pareto front, not a boolean.
    pf: bool = Field(True, description="Flag to save Pareto front", example=False)

    @field_validator("generations")
    @classmethod
    def check_generations(cls, v):
        if v > 1_000_000:
            raise ValueError("Too many generations, may exhaust memory")
        return v


class OptimizationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: np.ndarray = Field(..., description="Decision variables of the final population")
    F: np.ndarray = Field(..., description="Objective values of the final population")
    G: np.ndarray | None = Field(
        None, description="Constraint values for the final population"
    )
    CV: np.ndarray | None = Field(
        None, description="Aggregated constraint violation for the final population"
    )
    history: list[dict[str, np.ndarray]] | None = Field(
        None, description="History of algorithm's population"
    )

    @field_validator("X", "F", "G", "CV", mode="before")
    @classmethod
    def ensure_numpy_array(cls, v):
        return np.asarray(v) if v is not None else v

    @model_validator(mode="after")
    def post_init_checks(self):
        # Fallbacks for unconstrained problems
        if self.G is None:
            self.G = np.zeros((self.F.shape[0], 0))
        if self.CV is None:
            self.CV = np.zeros(self.F.shape[0])

        # Ensure CV is always strictly 1D
        self.CV = np.atleast_1d(self.CV).ravel()
        return self

    @cached_property
    def _is_feasible_mask(self) -> np.ndarray:
        return self.CV <= 0.0

    @cached_property
    def _feasible_indices(self) -> np.ndarray:
        return np.flatnonzero(self._is_feasible_mask)

    @cached_property
    def _pareto_indices(self) -> np.ndarray:
        if not self._is_feasible_mask.any():
            return np.array([], dtype=int)

        feasible_F = self.F[self._feasible_indices]
        nd_indices = NonDominatedSorting().do(feasible_F, only_non_dominated_front=True)
        return self._feasible_indices[nd_indices]

    @property
    def pareto_set(self) -> np.ndarray:
        return self.X[self._pareto_indices]

    @property
    def pareto_front(self) -> np.ndarray:
        return self.F[self._pareto_indices]

    @cached_property
    def last_generation_data(self) -> dict[str, np.ndarray] | None:
        """Extracts feasible points (both dominated and non-dominated) from the final generation."""
        if not self.history:
            return None

        # Access the last generation
        last_gen = self.history[-1]

        if last_gen.get("X") is None or last_gen.get("F") is None:
            return None

        X_last = last_gen["X"]
        F_last = last_gen["F"]

        # Handle CV
        CV_last = last_gen.get("CV")
        if CV_last is not None:
            CV_last = CV_last.ravel()
        else:
            CV_last = np.zeros(len(X_last))

        # Filter for feasibility
        feasible_mask = CV_last <= 0.0
        X_feas = X_last[feasible_mask]
        F_feas = F_last[feasible_mask]

        if X_feas.size == 0:
            return None

        return {
            "solutions": X_feas,
            "objectives": F_feas,
        }


class Minimizer:
    def __init__(self, problem, algorithm, config: MinimizerConfig):
        self.problem = problem
        self.algorithm = algorithm
        self.config = config

    def run(self) -> OptimizationResult:
        result: Result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=("n_gen", self.config.generations),
            seed=self.config.seed,
            save_history=self.config.save_history,
            verbose=self.config.verbose,
        )
        history_data = None
        if result.history is not None:
            history_data = [
                {
                    "X": algo.pop.get("X"),
                    "F": algo.pop.get("F"),
                    "G": algo.pop.get("G"),
                    "CV": algo.pop.get("CV"),
                }
                for algo in result.history
                if algo.pop is not None
            ]

        return OptimizationResult(
            X=result.X, F=result.F, G=result.G, CV=result.CV, history=history_data
        )
