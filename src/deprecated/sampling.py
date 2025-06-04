from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def sample_random_solutions(
    problem: Callable[[NDArray], NDArray],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    num_samples: int = 1000,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Generate random solutions within specified bounds and evaluate their objectives.

    Args:
        problem: Function that takes a decision vector and returns objective values
        lower_bounds: Sequence of lower bounds for each dimension
        upper_bounds: Sequence of upper bounds for each dimension
        num_samples: Number of random solutions to generate
        seed: Optional random seed for reproducibility

    Returns:
        tuple containing:
        - X: Array of decision vectors (shape: [num_samples, num_dimensions])
        - F: Array of objective vectors (shape: [num_samples, num_objectives])

    Raises:
        ValueError: If bounds sequences have mismatched lengths

    Example:
        >>> def dummy_problem(x): return np.array([x.sum(), x.max()])
        >>> X, F = sample_random_solutions(dummy_problem, [0, 0], [1, 1], 100)
        >>> X.shape
        (100, 2)
        >>> F.shape
        (100, 2)
    """
    # Input validation
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError(
            f"Bounds mismatch: lower_bounds (len {len(lower_bounds)}) "
            f"and upper_bounds (len {len(upper_bounds)}) must be same length"
        )

    # Create random number generator with optional seed
    rng = np.random.default_rng(seed)
    num_dim = len(lower_bounds)

    # Generate random samples
    X = rng.uniform(low=lower_bounds, high=upper_bounds, size=(num_samples, num_dim))

    # Evaluate all solutions (consider vectorization if problem supports it)
    F = np.array([problem(x) for x in X])

    return X, F
