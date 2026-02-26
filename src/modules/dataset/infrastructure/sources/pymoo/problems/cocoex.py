import cocoex
import numpy as np
from pydantic import BaseModel, Field
from pymoo.core.problem import Problem


class COCOBiObjectiveProblemConfig(BaseModel):
    """
    Problem specification for optimization algorithms.
    """

    problem_id: int = Field(
        ..., ge=1, description="The problem indices in the coco framework"
    )
    n_var: int = Field(
        ...,
        ge=1,
        description="Number of decision variables in the optimization problem",
    )
    n_obj: int = Field(2, frozen=True, description="Number of objectives")
    n_constr: int = Field(0, frozen=True, description="Number of constraints")


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

        suite_options = (
            f"dimensions:{dimensions} "
            f"instance_indices:{instance_indices} "
            f"function_indices:{function_indices}"
        )

        suite = cocoex.Suite(
            problem_name,
            "",
            suite_options,
        )

        problem = suite.get_problem(0)
        return problem
