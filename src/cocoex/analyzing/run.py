import matplotlib.pyplot as plt
import numpy as np
from interpolators.linear import LinearInterpolator
from recommender import ObjectivePreferences, ParetoAnalyzer
from cocoex.utils.similarities import cosine_similarity
from utils.data_manager import ParetoDataManager, normalize_to_hypercube


def main():
    # Load saved data
    manager = ParetoDataManager()

    raw_data = manager.load(
        filename="pareto_data.npz", normalize=normalize_to_hypercube
    )

    # Access individual components
    pareto_set = raw_data.pareto_set
    pareto_front = raw_data.pareto_front

    print(pareto_set)
    # ! check the normalization validity becuase the data has negative values

    # # Create and fit the interpolator
    # interpolator = LinearInterpolator()
    # interpolator.fit(solutions=pareto_set_normalized, pareto=pareto_front_normalized)

    # # Initialize analyzer with Pareto-optimal solutions
    # analyzer = ParetoAnalyzer(
    #     solutions=pareto_set_normalized,
    #     front=pareto_front_normalized,
    #     interpolator=interpolator,
    #     similarity_metric=cosine_similarity,
    #     sort_by_obj=0,  # Sort by first objective (e.g., time)
    # )

    # # Define user's desired objective trade-off (e.g., 70% weight on first objective)
    # user_preference = ObjectivePreferences(time_weight=0.8, energy_weight=0.2)

    # # Find best matching solutions
    # candidate_indices = analyzer.recommend_from_top(user_preference, k=2)
    # print(f"Top candidate solutions at indices: {candidate_indices}")

    # Calculate interpolation position
    # interpolation_alpha = analyzer.calculate_interpolation_parameter(user_preference)
    # print(f"Interpolation position: α={interpolation_alpha:.2f}")

    # # Generate interpolated solution
    # optimized_solution = interpolator(interpolation_alpha)
    # print(f"Optimized solution: {optimized_solution}")

    # # Verify preference alignment
    # alignment_scores = analyzer.calculate_cosine_similarity(user_preference)
    # print(f"Maximum alignment score: {np.max(alignment_scores):.2f}")

    # # --- Geodesic Interpolation ---
    # geodesic_interp = GeodesicInterpolator (pareto_set)
    # geodesic_solution = geodesic_interp(alpha)
    # print(f"Geodesic Interpolation (α={alpha}):", geodesic_solution)

    # # --- Neural Network Interpolation ---
    # nn_interp = NNInterpolator(input_dim=pareto_set.shape[1])
    # alphas_train = np.linspace(0, 1, len(pareto_set))
    # nn_interp.fit(alphas_train, pareto_set, epochs=5000)
    # nn_solution = nn_interp.predict(alpha)
    # print(f"Neural Net Interpolation (α={alpha}):", nn_solution)

    # # --- Visualize Interpolated Solutions ---
    # plt.figure(figsize=(10, 6))
    # plt.scatter(pareto_set[:, 0], pareto_set[:, 1], label="Pareto Set")
    # plt.scatter(linear_solution[0], linear_solution[1], c="red", s=100, label="Linear")
    # plt.scatter(
    #     geodesic_solution[0], geodesic_solution[1], c="green", s=100, label="Geodesic"
    # )
    # plt.scatter(
    #     nn_solution[0][0], nn_solution[0][1], c="purple", s=100, label="Neural Net"
    # )
    # plt.legend()
    # plt.title("Interpolation Comparison")
    # plt.show()

    # Visualize trade-off landscape

    # plt.scatter(
    #     analyzer.normalized_objectives[:, 0],
    #     analyzer.normalized_objectives[:, 1],
    #     c=analyzer.calculate_preference_alignment(user_preference),
    # )
    # plt.xlabel("Normalized Cost")
    # plt.ylabel("Normalized Performance")
    # plt.title("Design Preference Alignment")
    # plt.colorbar(label="Preference Match Score")
    # plt.show()


if __name__ == "__main__":
    main()
