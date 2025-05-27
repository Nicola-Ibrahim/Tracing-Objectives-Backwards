from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_pareto_set(
    x_opt1: NDArray[np.float64], x_opt2: NDArray[np.float64], num_points: int = 100
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a Pareto set by linear interpolation between two optimal solutions.

    For convex bi-objective problems like Sphere/Sphere, the Pareto set is
    the straight line connecting the individual optima of each objective.

    Args:
        x_opt1: Optimal solution for first objective (shape: [n_dimensions,])
        x_opt2: Optimal solution for second objective (shape: [n_dimensions,])
        num_points: Number of points to generate along the Pareto set

    Returns:
        Tuple containing:
        - alpha: Interpolation parameters (shape: [num_points,])
        - pareto_set: Pareto optimal solutions (shape: [num_points, n_dimensions])

    Raises:
        ValueError: If input solutions have different dimensions

    Example:
        >>> x1 = np.array([0, 0])
        >>> x2 = np.array([1, 1])
        >>> alpha, ps = generate_pareto_set(x1, x2, num_points=3)
        >>> ps
        array([[1. , 1. ],
               [0.5, 0.5],
               [0. , 0. ]])
    """
    if x_opt1.shape != x_opt2.shape:
        raise ValueError(
            f"Dimension mismatch between x_opt1 {x_opt1.shape} and x_opt2 {x_opt2.shape}"
        )

    alpha = np.linspace(0, 1, num_points)
    pareto_set = np.array([a * x_opt1 + (1 - a) * x_opt2 for a in alpha])
    return alpha, pareto_set


def evaluate_pareto_front(
    problem: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    pareto_set: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate the Pareto front for a set of Pareto optimal solutions.

    Args:
        problem: Function that maps decision vectors to objective vectors
        pareto_set: Array of Pareto optimal solutions (shape: [n_solutions, n_dimensions])

    Returns:
        Array of objective vectors (shape: [n_solutions, n_objectives])

    Example:
        >>> def dummy_problem(x): return np.array([x.sum(), x.max()])
        >>> ps = np.array([[0,0], [0.5,0.5], [1,1]])
        >>> evaluate_pareto_front(dummy_problem, ps)
        array([[0. , 0. ],
               [1. , 0.5],
               [2. , 1. ]])
    """
    return np.array([problem(x) for x in pareto_set])
