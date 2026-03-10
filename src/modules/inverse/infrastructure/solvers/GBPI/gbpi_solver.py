from typing import Any

import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from .....modeling.infrastructure.estimators.deterministic.rbf import (
    RBFEstimator,
    RBFEstimatorParams,
)
from ....domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
    InverseSolverResult,
)
from .sampling.coherent import CoherentSampling
from .sampling.extrapolation import ExtrapolationSampling
from .sampling.incoherent import IncoherentSampling


class GBPIInverseSolver(AbstractInverseMappingSolver):
    """
    Concrete implementation of the Geometrically Bounded Probabilistic Inversion (GBPI) framework.
    It explicitly owns its specific mathematical state and elegantly routes queries
    to prevent multi-modal averaging (the Centroid Trap) and handle out-of-bounds targets.
    """

    def __init__(
        self,
        coherence_n_neighbors: int = 5,
        trust_radius: float = 0.05,
        concentration_factor: float = 100,
    ):
        """
        Initializes the GBPISolver.

        Args:
            coherence_n_neighbors: How many local X-space neighbors to use when calculating D_max (tau).
            trust_radius: The trust-region radius for the Gradient Descent optimization.
            concentration_factor: The concentration factor for the Dirichlet distribution.
        """
        self.coherence_n_neighbors = coherence_n_neighbors
        self.trust_radius = trust_radius
        self.concentration_factor = concentration_factor

        self.X = None
        self.tau = None
        self.mesh = None
        self.objective_knn = None
        self.forward_estimator = None

    def type(self):
        return "GBPI"

    def _ensure_fitted(self):
        """Ensures all necessary models and thresholds are trained before generating."""
        if (
            self.forward_estimator is None
            or self.X is None
            or self.tau is None
            or self.mesh is None
            or self.objective_knn is None
        ):
            raise ValueError("The solver has not been trained yet.")

    def _evaluate_coherence(
        self, vertices_indices: list[int]
    ) -> tuple[bool, list[float]]:
        """
        Enforces the coherence domain rule (D_max).
        Checks if the X-space distance between the performance anchors is safe for interpolation.

        Args:
            vertices_indices: List of dataset indices representing the 3 anchor points.

        Returns:
            is_coherent: True if all pairwise distances <= tau (D_max).
            pairwise_dists: List of the actual calculated Euclidean distances.
        """
        triangle_vertices = self.X[vertices_indices]

        if len(triangle_vertices) < 2:
            return True, []

        diffs = triangle_vertices[:, None, :] - triangle_vertices[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        i, j = np.triu_indices(len(triangle_vertices), k=1)
        pairwise_dists = dists[i, j].tolist()

        is_coherent = bool(np.all(np.array(pairwise_dists) <= self.tau))
        return is_coherent, pairwise_dists

    def _locate_in_mesh(self, target: np.ndarray) -> tuple[bool, list[int], np.ndarray]:
        """
        STRICTLY checks if the target is physically surrounded by known data in Y-space.

        Args:
            target: The user's desired 2D objective performance.

        Returns:
            is_inside: True if a bounding triangle is found.
            vertices_indices: List of the 3 anchor indices (empty if outside).
            weights: Barycentric weights determining the target's position (empty if outside).
        """
        simplex_idx = self.mesh.find_simplex(target)[0]

        if simplex_idx != -1:
            # Target is INSIDE the convex hull
            transform = self.mesh.transform[simplex_idx]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]
            vertices_indices = self.mesh.simplices[simplex_idx].tolist()
            return True, vertices_indices, weights
        else:
            # Target is OUTSIDE the convex hull
            return False, [], np.array([])

    def _get_nearest_neighbor(self, target: np.ndarray) -> tuple[list[int], np.ndarray]:
        """
        Uses the global Y-space KNN to find the single absolute closest point for extrapolation.
        """
        _, indices = self.objective_knn.kneighbors(target)
        closest_vertex_idx = int(indices[0][0])

        return [closest_vertex_idx], np.array([1.0])

    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        """
        The main traffic controller of the GBPI framework.
        Routes the target to Interpolation, Trust-Region Optimization, or Extrapolation.
        """
        self._ensure_fitted()

        target_y = np.asarray(target_y).reshape(1, -1)
        if target_y.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        # --- Step 1: Locate the target in relation to the known data manifold ---
        is_inside_mesh, vertices_indices, weights = self._locate_in_mesh(target_y)

        is_coherent = False
        anchor_distances = []
        pathway = "unknown"
        final_vertices_X = None
        final_weights = None

        # --- Step 2: Route to the correct mathematical pathway ---
        if is_inside_mesh:
            # The target is surrounded by data. Check if those data points are physically compatible.
            is_coherent, anchor_distances = self._evaluate_coherence(vertices_indices)

            if is_coherent:
                # PATHWAY A: Safe to interpolate.
                # The anchors share an engineering strategy. Use Dirichlet to generate diverse options.
                pathway = "coherent"
                final_vertices_X = self.X[vertices_indices]
                final_weights = weights

                candidates_X = CoherentSampling(
                    concentration_factor=self.concentration_factor
                ).sample(
                    vertices_X=final_vertices_X,
                    weights=final_weights,
                    n_samples=n_samples,
                )

            else:
                # PATHWAY B1: Incoherent Triangle (Centroid Trap avoidance).
                # Anchors use completely different strategies. Do NOT interpolate.
                # Find the single closest anchor to use as a base for Gradient Descent.
                pathway = "incoherent"

                best_idx_position = np.argmax(weights)
                single_best_vertex_idx = [vertices_indices[best_idx_position]]

                final_vertices_X = self.X[single_best_vertex_idx]
                final_weights = np.array([1.0])

                candidates_X = IncoherentSampling(
                    forward_estimator=self.forward_estimator,
                    target_y=target_y,
                    trust_radius=self.trust_radius,
                ).sample(
                    vertices_X=final_vertices_X,
                    weights=final_weights,
                    n_samples=n_samples,
                )
        else:
            # PATHWAY B2: Out of Bounds (Extrapolation).
            # The user asked for performance beyond the known dataset.
            # Use global KNN to find the single best jumping-off point for Gradient Descent.
            pathway = "extrapolation"

            nn_indices, nn_weights = self._get_nearest_neighbor(target_y)
            vertices_indices = (
                nn_indices  # Update metadata to reflect the single point used
            )

            final_vertices_X = self.X[nn_indices]
            final_weights = nn_weights

            candidates_X = ExtrapolationSampling(
                forward_estimator=self.forward_estimator,
                target_y=target_y,
                trust_radius=self.trust_radius,
            ).sample(
                vertices_X=final_vertices_X,
                weights=final_weights,
                n_samples=n_samples,
            )

        # --- Step 3: Predict candidates objectives to detect Pareto Sag ---
        candidates_y = self.forward_estimator.predict(candidates_X)

        return InverseSolverResult(
            candidates_X=candidates_X,
            candidates_y=candidates_y,
            metadata={
                "pathway": pathway,
                "is_simplex_found": is_inside_mesh,
                "is_coherent": is_coherent,
                "anchor_distances": anchor_distances,
                "vertices_indices": vertices_indices,
            },
        )

    def _train_forward_estimator(self, X: np.ndarray, y: np.ndarray):
        """Trains the RBF surrogate to act as a fast, differentiable physical validator."""
        params = RBFEstimatorParams(n_neighbors=7, kernel="thin_plate_spline")
        estimator = RBFEstimator(params)
        estimator.fit(X, y)
        return estimator

    def _calculate_coherence_threshold(self, X: np.ndarray):
        """
        KNN #1: The X-Space Physics Checker (Offline).
        Calculates D_max (tau) based on the 95th percentile of local X-space distances.
        """
        nn = NearestNeighbors(n_neighbors=self.coherence_n_neighbors + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        self.tau = float(np.percentile(distances[:, 1:], 95))

    def _build_mesh(self, y: np.ndarray):
        """Builds the 2D Delaunay triangulation mesh in Objective (Y) space."""
        self.mesh = Delaunay(y)

    def _build_y_knn(self, y: np.ndarray):
        """
        KNN #2: The Y-Space Extrapolation Router (Offline).
        Builds the spatial index for fast O(log N) out-of-bounds nearest neighbor lookups.
        Hardcoded to 1 neighbor because Gradient Descent requires a single base design.
        """
        self.objective_knn = NearestNeighbors(n_neighbors=1)
        self.objective_knn.fit(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Executes Phase 0: System Initialization & Pre-Processing."""
        self.forward_estimator = self._train_forward_estimator(X, y)
        self._calculate_coherence_threshold(X)
        self._build_mesh(y)
        self.X = X
        self._build_y_knn(y)

    def history(self) -> dict[str, Any]:
        """Returns the offline training artifacts."""
        return {
            "tau": self.tau,
        }
