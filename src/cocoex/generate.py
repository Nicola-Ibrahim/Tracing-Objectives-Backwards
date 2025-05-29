from analyzing.interpolators.linear import LinearInterpolator
from generating.optimization import find_optima
from generating.problem import get_problem, get_problem_parameters
from generating.sampling import sample_random_solutions
from generating.visualization import plot_pareto_front
from cocoex.utils.data_manager import save_pareto_data
from utils.pareto import evaluate_pareto_front, generate_pareto_set


def main():
    # Initialize problem
    problem = get_problem()
    params = get_problem_parameters(problem)
    dim = params["dim"]
    lower_bounds = params["lower_bounds"]
    upper_bounds = params["upper_bounds"]

    # Find optima
    x_opt1, x_opt2 = find_optima(problem, dim, lower_bounds, upper_bounds)

    # Generate Pareto set/front
    alpha, pareto_set = generate_pareto_set(x_opt1, x_opt2)
    pareto_front = evaluate_pareto_front(problem, pareto_set)

    # Save Pareto data
    save_pareto_data(pareto_set, pareto_front, problem_name="F1_Sphere_Sphere")

    # Sample solutions
    # X, F = sample_random_solutions(problem, lower_bounds, upper_bounds)

    # # Visualize
    # plot_pareto_front(F, pareto_front)


if __name__ == "__main__":
    main()
