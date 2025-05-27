import cocoex


def get_problem():
    """Initialize the bbob-biobj F1 (Sphere/Sphere) problem."""
    suite = cocoex.Suite(
        "bbob-biobj",
        "",
        "year: 2016 function_indices:1 dimensions:2 instance_indices:1",
    )
    problem = suite.get_problem(0)
    return problem


def get_problem_parameters(problem):
    """Extract problem parameters (dimension, bounds)."""
    return {
        "dim": problem.dimension,
        "lower_bounds": problem.lower_bounds,
        "upper_bounds": problem.upper_bounds,
    }
