from typing import Callable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def find_optima(
    problem: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dim: int,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    opt_method: str = "L-BFGS-B",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find the optimal solutions for individual objectives in a bi-objective optimization problem.

    Args:
        problem: Function that takes a decision vector and returns objective values.
                 Must return at least two objectives.
        dim: Dimension of the decision space
        lower_bounds: Sequence of lower bounds for each dimension
        upper_bounds: Sequence of upper bounds for each dimension
        opt_method: Optimization method to use (default: "L-BFGS-B")

    Returns:
        Tuple containing:
        - x_opt1: Optimal solution for first objective (shape: [dim,])
        - x_opt2: Optimal solution for second objective (shape: [dim,])

    Raises:
        ValueError: If bounds have incorrect length or problem returns <2 objectives
        RuntimeError: If optimization fails to converge

    Example:
        >>> def problem(x): return np.array([x@x, (x-1)@(x-1)])
        >>> x1, x2 = find_optima(problem, dim=2, lower_bounds=[-5,-5], upper_bounds=[5,5])
        >>> x1.round(2)
        array([0., 0.])
        >>> x2.round(2)
        array([1., 1.])
    """
    # Input validation
    if len(lower_bounds) != dim or len(upper_bounds) != dim:
        raise ValueError(f"Bounds length must match problem dimension {dim}")

    test_output = problem(np.zeros(dim))
    if len(test_output) < 2:
        raise ValueError("Problem must return at least two objectives")

    bounds = list(zip(lower_bounds, upper_bounds))

    def optimize_objective(obj_index: int) -> NDArray[np.float64]:
        """Helper function to optimize individual objectives"""
        res = minimize(
            fun=lambda x: problem(x)[obj_index],
            x0=np.zeros(dim),
            bounds=bounds,
            method=opt_method,
        )

        if not res.success:
            raise RuntimeError(f"Optimization failed for f{obj_index+1}: {res.message}")
        return res.x

    try:
        x_opt1 = optimize_objective(0)
        x_opt2 = optimize_objective(1)
    except Exception as e:
        raise RuntimeError("Failed to find optimal solutions") from e

    return x_opt1, x_opt2
