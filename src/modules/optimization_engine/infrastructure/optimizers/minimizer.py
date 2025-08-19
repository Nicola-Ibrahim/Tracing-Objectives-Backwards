from pydantic import BaseModel, Field, field_validator
from pymoo.optimize import minimize

from ...domain.generation.interfaces.base_optimizer import ProblemAwareOptimizer
from .result import OptimizationResult


class MinimizerConfig(BaseModel):
    generations: int = Field(
        ..., gt=0, description="Number of generations must be > 0", example=100
    )
    seed: int = Field(
        42, ge=0, description="Random seed must be non-negative", example=42
    )
    save_history: bool = Field(
        ..., description="Flag to save optimization history", example=True
    )
    verbose: bool = Field(..., description="Flag for verbose output", example=False)
    pf: bool = Field(True, description="Flag to save Pareto front", example=False)

    @field_validator("generations")
    def check_generations(cls, v):
        if v > 1_000_000:
            raise ValueError("Too many generations, may exhaust memory")
        return v

    class Config:
        arbitrary_types_allowed = True


class Minimizer(ProblemAwareOptimizer):
    def run(self) -> OptimizationResult:
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            generations=self.config.generations,
            seed=self.config.seed,
            save_history=self.config.save_history,
            verbose=self.config.verbose,
            pf=self.config.pf,
        )

        # --- FIX STARTS HERE ---
        converted_history = []
        if result.history is not None:
            for algo_in_history in result.history:
                # The population data is stored in the algorithm object's 'pop' attribute
                pop = algo_in_history.pop

                # Check if the population is valid and extract the data
                if pop is not None:
                    entry_dict = {
                        "X": pop.get("X"),
                        "F": pop.get("F"),
                        "G": pop.get("G"),
                        "CV": pop.get("CV"),
                    }
                    converted_history.append(entry_dict)

        return OptimizationResult(
            X=result.X, F=result.F, G=result.G, CV=result.CV, history=converted_history
        )
