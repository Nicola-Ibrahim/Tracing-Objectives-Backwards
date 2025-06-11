import cocoex
import numpy as np
from cocoex import Problem as COCOProblem
from pydantic import BaseModel, Field

from ...domain.interfaces.base_problem import BaseProblem


def get_coco_problem(
    problem_name: str = "bbob-biobj",
    function_indices: int = 1,
    instance_indices: int = 1,
    dimensions: int = 2,
) -> COCOProblem:
    """Initialize a COCO BBOB-BIOBJ problem with specified configuration."""

    # Validate function index for bbob-biobj
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
        problem_name,  # suite_name
        "",  # suite_instance
        suite_options,  # suite_options (must only use valid keys!)
    )

    problem = suite.get_problem(0)
    return problem


class BiObjProblemConfig(BaseModel):
    """
    Problem specification for optimization algorithms.
    Contains the target distance for the mission and the vehicle configuration.
    """

    problem_id: int = Field(
        5, ge=1, description="The problem indices in the coco framework"
    )

    n_var: int = Field(
        ...,
        ge=1,
        description="Number of decision variables in the optimization problem",
    )

    n_obj: int = Field(
        2, ge=2, description="Number of objectives in the optimization problem"
    )

    n_constr: int = Field(
        0, ge=0, description="Number of constraints in the optimization problem"
    )

    xl: list[float] = Field(..., description="Lower bounds for decision variables")
    xu: list[float] = Field(..., description="Upper bounds for decision variables")


class COCOBiObjectiveProblem(BaseProblem):
    """
    Adapter for COCO bi-objective problems to pymoo's Problem interface.

    This class wraps a COCO problem instance and adapts it to work with pymoo's
    optimization framework while maintaining the BaseProblem interface.

    """

    def __init__(self, problem: COCOProblem, config: BiObjProblemConfig):
        """
        Initialize the BiObjectiveProblem with a COCO problem instance.
        Args:
            problem: A COCO problem instance that implements the callable interface.
            lower_bounds: Lower bounds for the decision variables.
            upper_bounds: Upper bounds for the decision variables.
        """
        self.coco_problem = problem

        super().__init__(
            n_var=config.n_var,
            n_obj=config.n_obj,
            n_constr=config.n_constr,
            xl=np.array(config.xl),
            xu=np.array(config.xu),
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate the COCO problem for multiple solutions"""
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            result = self.coco_problem(x)
            F[i, :] = result[: self.n_obj]
        out["F"] = F
