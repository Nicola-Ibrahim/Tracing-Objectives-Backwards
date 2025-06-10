from pydantic import BaseModel, Field, field_validator
from pymoo.optimize import minimize

from ...domain.interfaces.optimizer import BaseOptimizer
from .result import OptimizationResult


class MinimizerConfig(BaseModel):
    generations: int = Field(
        ..., gt=0, description="Number of generations must be > 0"
    )  # Number of generations for the optimization
    seed: int = Field(
        ..., ge=0, description="Random seed must be non-negative"
    )  # Random seed for reproducibility
    save_history: bool = False  # Flag to save optimization history
    verbose: bool = True  # Verbosity flag for logging
    pf: bool = False  # Flag to save Pareto front

    class Config:
        frozen = True  # make it immutable

    @field_validator("generations")
    def check_generations(cls, v):
        if v > 1_000_000:
            raise ValueError("Too many generations, may exhaust memory")
        return v


class Minimizer(BaseOptimizer):
    def run(self) -> OptimizationResult:
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            **self.config.model_dump(),
        )
        return OptimizationResult(
            X=result.X, F=result.F, G=result.G, CV=result.CV, history=result.history
        )
