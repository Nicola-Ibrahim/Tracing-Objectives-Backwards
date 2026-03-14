import cocoex
import numpy as np
from pydantic import BaseModel, Field, model_validator
from pymoo.core.problem import Problem


class COCOBiObjectiveProblemConfig(BaseModel):
    """
    Problem specification for optimization algorithms.
    """

    problem_id: int = Field(
        5, ge=1, description="The problem indices in the coco framework"
    )
    instance_indices: int = Field(
        1,
        ge=1,
        description="The instance indices in the coco framework (affects range/shifts)",
    )
    n_var: int = Field(
        2,
        ge=1,
        description="Number of decision variables in the optimization problem",
    )
    n_obj: int = Field(2, frozen=True, description="Number of objectives")
    n_constr: int = Field(0, frozen=True, description="Number of constraints")

    @model_validator(mode="after")
    def validate_dimension(self) -> "COCOBiObjectiveProblemConfig":
        allowed_dims = [2, 3, 5, 10, 20, 40]
        if self.n_var not in allowed_dims:
            raise ValueError(
                f"Dimension {self.n_var} not supported by 'bbob-biobj' suite. "
                f"Supported dimensions are {allowed_dims}."
            )
        return self


class COCOBiObjectiveProblem(Problem):
    """
    Adapter for COCO bi-objective problems to pymoo's Problem interface.
    """

    def __init__(self, config: COCOBiObjectiveProblemConfig):
        """
        Initialize the BiObjectiveProblem with a COCO problem instance.
        """
        self.coco_problem = self._get_problem(
            problem_name="bbob-biobj",
            function_indices=config.problem_id,
            instance_indices=config.instance_indices,
            dimensions=config.n_var,
        )

        super().__init__(
            n_var=config.n_var,
            n_obj=config.n_obj,
            n_constr=config.n_constr,
            xl=np.array(self.coco_problem.lower_bounds),
            xu=np.array(self.coco_problem.upper_bounds),
        )

        # This correctly prevents pickling errors during multiprocessing in pymoo
        self.exclude_from_serialization = ["coco_problem"]

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate the COCO problem for multiple solutions"""
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            result = self.coco_problem(x)
            F[i, :] = result[: self.n_obj]
        out["F"] = F

    def _get_problem(
        self,
        problem_name: str = "bbob-biobj",
        function_indices: int = 1,
        instance_indices: int = 1,
        dimensions: int = 2,
    ) -> cocoex.Problem:
        """Initialize a COCO BBOB-BIOBJ problem with specified configuration."""

        if problem_name == "bbob-biobj" and not (1 <= function_indices <= 55):
            raise ValueError(
                f"`function_indices` must be between 1 and 55 for suite '{problem_name}', got {function_indices}."
            )

        suite_options = f"dimensions:{dimensions} function_indices:{function_indices}"

        suite = cocoex.Suite(
            problem_name,
            f"instance_indices:{instance_indices}",
            suite_options,
        )

        problem = suite.get_problem(0)
        return problem
