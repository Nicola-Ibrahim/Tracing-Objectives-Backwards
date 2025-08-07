import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity

# Generate synthetic 2D Pareto front points (non-convex shape)
np.random.seed(1)
theta = np.linspace(0, np.pi, 30)
pareto_x = np.cos(theta) + 0.05 * np.random.randn(len(theta))
pareto_y = np.sin(theta) + 0.05 * np.random.randn(len(theta))
pareto_points = np.vstack([pareto_x, pareto_y]).T

# Normalize to [0, 1]
pareto_min = pareto_points.min(axis=0)
pareto_max = pareto_points.max(axis=0)
pareto_norm = (pareto_points - pareto_min) / (pareto_max - pareto_min)

# Target points
target_inside_circle = pareto_norm[12] + np.array([0.03, -0.01])
target_outside = np.array([1.1, 0.1])
target_near_hull = np.array([0.95, 0.5])

# Prepare figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# ========== LCNF with Spheres Plot ==========
radius = 0.08
for p in pareto_norm:
    circle = plt.Circle(p, radius, color="blue", alpha=0.1)
    axs[0].add_patch(circle)

axs[0].scatter(*pareto_norm.T, c="black", label="Pareto front")
axs[0].scatter(*target_inside_circle, c="green", label="Target (Inside)", edgecolor="k")
axs[0].scatter(*target_outside, c="red", label="Target (Outside)", edgecolor="k")
axs[0].set_title("LCNF - Local Spheres")
axs[0].set_xlim(-0.2, 1.4)
axs[0].set_ylim(-0.2, 1.4)
axs[0].legend()
axs[0].set_aspect("equal")

# ========== KDE Plot ==========
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

kde = KernelDensity(bandwidth=0.07)
kde.fit(pareto_norm)
scores = np.exp(kde.score_samples(grid_points)).reshape(100, 100)

axs[1].imshow(scores.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", alpha=0.7)
axs[1].scatter(*pareto_norm.T, c="black")
axs[1].scatter(*target_inside_circle, c="green", edgecolor="k")
axs[1].scatter(*target_outside, c="red", edgecolor="k")
axs[1].set_title("KDE - Density Estimation")
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_aspect("equal")

# ========== Convex Hull + Buffer Plot ==========
hull = ConvexHull(pareto_norm)
for simplex in hull.simplices:
    axs[2].plot(pareto_norm[simplex, 0], pareto_norm[simplex, 1], "b-")

# Add buffer as transparent disks (simplified buffer region)
for i in range(len(pareto_norm)):
    axs[2].add_patch(plt.Circle(pareto_norm[i], radius / 2, color="red", alpha=0.05))

axs[2].scatter(*pareto_norm.T, c="black")
axs[2].scatter(*target_near_hull, c="orange", label="Target (Buffered)", edgecolor="k")
axs[2].scatter(*target_outside, c="red", label="Target (Outside)", edgecolor="k")
axs[2].set_title("Convex Hull + Buffer")
axs[2].set_xlim(0, 1.4)
axs[2].set_ylim(0, 1.2)
axs[2].legend()
axs[2].set_aspect("equal")

plt.tight_layout()
plt.show()
