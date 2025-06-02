from utils.archivers.npz import ParetoDataManager

from generate.optimization import find_optima
from generate.pareto import evaluate_pareto_front, generate_pareto_set
from generate.problem import get_problem
from generate.sampling import sample_random_solutions
from generate.visualization import plot_pareto_visualizations


def main():
    # Initialize problem
    problem = get_problem()
    dim = problem.dimension
    lower_bounds = problem.lower_bounds
    upper_bounds = problem.upper_bounds

    print(problem.id)
    print(problem.name)

    # Find optima solutions
    f1_optimal, f2_optimal = find_optima(problem, dim, lower_bounds, upper_bounds)

    print(f"f1_optimal: {f1_optimal}")
    print(f"f2_optimal: {f2_optimal}")

    # Generate Pareto set & front
    alpha, pareto_set = generate_pareto_set(f1_optimal, f2_optimal)
    pareto_front = evaluate_pareto_front(problem, pareto_set)

    # Save Pareto data
    data_manager = ParetoDataManager()
    data_manager.save(
        pareto_set=pareto_set,
        pareto_front=pareto_front,
        problem_name="F1_Sphere_Sphere",
        metadata={"alpha": alpha, "problem_name": problem.name, "function": problem.id},
    )

    # Sample solutions
    rnd_solution_set, rnd_solution_objectives = sample_random_solutions(
        problem, lower_bounds, upper_bounds
    )

    # Visualize
    plot_pareto_visualizations(
        rnd_solution_set,
        rnd_solution_objectives,
        pareto_set,
        pareto_front,
        f1_optimal,
        f2_optimal,
    )


if __name__ == "__main__":
    main()
