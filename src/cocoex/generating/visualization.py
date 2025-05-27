import matplotlib.pyplot as plt


def plot_pareto_front(F, pareto_front):
    """Visualize the Pareto front and sampled solutions."""
    plt.figure(figsize=(10, 6))
    plt.scatter(F[:, 0], F[:, 1], s=5, alpha=0.5, label="Sampled Solutions")
    plt.scatter(
        pareto_front[:, 0], pareto_front[:, 1], s=20, c="red", label="True Pareto Front"
    )
    plt.xlabel("$f_1$", fontsize=12)
    plt.ylabel("$f_2$", fontsize=12)
    plt.title("Pareto Front for F1: Sphere/Sphere", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
