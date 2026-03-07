from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, distance_matrix
from sklearn.neighbors import NearestNeighbors

from .....modeling.infrastructure.estimators.deterministic.rbf import (
    RBFEstimator,
    RBFEstimatorParams,
)
from ....domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
    InverseSolverResult,
)
from .sampling.dirichlet import DirichletSampling
from .sampling.gd import GradientDescentSampling


class TDAGBPIInverseSolver(AbstractInverseMappingSolver):
    """
    Advanced implementation of the GBPI framework using Topological Data Analysis (TDA).
    Replaces the Euclidean threshold with a Vietoris-Rips structural graph check
    to explicitly detect non-linear manifolds and topological holes.
    """

    def __init__(
        self,
        coherence_n_neighbors: int = 5,
        trust_radius: float = 0.05,
        concentration_factor: float = 100,
        hole_penalty_factor: float = 1.5,  # Ratio to detect a topological hole
    ):
        """
        Initializes the TDA-GBPISolver.

        Args:
            coherence_n_neighbors: How many local X-space neighbors to use when determining the TDA linking radius (epsilon).
            trust_radius: The trust-region radius for the gradient descent optimization.
            concentration_factor: The concentration factor for the Dirichlet distribution.
            hole_penalty_factor: The multiplier threshold. If the shortest topological path is
                                 this many times longer than the straight line, a hole is detected.
        """
        self.coherence_n_neighbors = coherence_n_neighbors
        self.trust_radius = trust_radius
        self.concentration_factor = concentration_factor
        self.hole_penalty_factor = hole_penalty_factor

        self.X = None
        self.mesh = None
        self.objective_knn = None
        self.forward_estimator = None

        # TDA Specific State
        self.vr_epsilon = None  # The Vietoris-Rips linking radius
        self.vr_graph = None  # The structural 1-skeleton graph of the dataset

    def type(self):
        return "TDA-GBPI"

    def _ensure_fitted(self):
        """Ensures all necessary models and topological structures are trained."""
        if (
            self.forward_estimator is None
            or self.X is None
            or self.vr_graph is None
            or self.mesh is None
            or self.objective_knn is None
        ):
            raise ValueError("The solver has not been trained yet.")

    def _build_topological_complex(self, X: np.ndarray):
        """
        Builds the 1-skeleton of the Vietoris-Rips complex for the dataset.
        This maps the "shape" of the data and finds connected physical components.
        """
        # 1. Find the characteristic local scale (epsilon) based on neighbors
        nn = NearestNeighbors(n_neighbors=self.coherence_n_neighbors + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        self.vr_epsilon = float(np.percentile(distances[:, 1:], 95))

        # 2. Build the adjacency matrix (connect points only if distance <= epsilon)
        dist_mat = distance_matrix(X, X)
        adjacency = np.where(dist_mat <= self.vr_epsilon, dist_mat, 0)

        # 3. Store as a sparse graph for lightning-fast topological traversal
        self.vr_graph = csr_matrix(adjacency)

    def _evaluate_coherence(
        self, vertices_indices: list[int]
    ) -> tuple[bool, list[float]]:
        """
        Enforces TDA Coherence.
        Checks if the straight Euclidean interpolation line cuts across a topological hole
        by comparing it to the shortest geodesic path traveling safely on the Vietoris-Rips graph.
        """
        triangle_vertices = self.X[vertices_indices]

        if len(triangle_vertices) < 2:
            return True, []

        is_coherent = True
        anchor_distances = []

        # Check all pairs of the triangle edges (A-B, B-C, A-C)
        for i in range(len(vertices_indices)):
            for j in range(i + 1, len(vertices_indices)):
                idx_A = vertices_indices[i]
                idx_B = vertices_indices[j]

                # 1. Calculate raw Euclidean distance (the unsafe straight line)
                euclidean_dist = np.linalg.norm(self.X[idx_A] - self.X[idx_B])
                anchor_distances.append(euclidean_dist)

                # 2. Calculate the Topological Geodesic distance (the safe structural path)
                geodesic_dist = dijkstra(
                    csgraph=self.vr_graph,
                    directed=False,
                    indices=idx_A,
                    return_predecessors=False,
                )[idx_B]

                # 3. The TDA Hole Detection Logic
                # Disconnected components trigger immediate failure.
                if np.isinf(geodesic_dist):
                    is_coherent = False
                    break

                # If the safe path is significantly longer than the straight line,
                # the straight line is cutting across an empty physical void.
                if geodesic_dist > (euclidean_dist * self.hole_penalty_factor):
                    is_coherent = False
                    break

        return is_coherent, anchor_distances

    def _locate_in_mesh(self, target: np.ndarray) -> tuple[bool, list[int], np.ndarray]:
        """
        STRICTLY checks if the target is physically surrounded by known data in Y-space.
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
        The main traffic controller. Routes the target to Interpolation, Trust-Region
        Optimization, or Extrapolation based on Delaunay position and TDA Coherence.
        """
        self._ensure_fitted()

        target_y = np.asarray(target_y).reshape(1, -1)
        if target_y.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        # --- Step 1: Locate the target ---
        is_inside_mesh, vertices_indices, weights = self._locate_in_mesh(target_y)

        is_coherent = False
        anchor_distances = []
        pathway = "unknown"
        final_vertices_X = None
        final_weights = None

        # --- Step 2: Route to the correct mathematical pathway ---
        if is_inside_mesh:
            # Check TDA coherence to see if the triangle crosses a structural hole
            is_coherent, anchor_distances = self._evaluate_coherence(vertices_indices)

            if is_coherent:
                # PATHWAY A: Safe to interpolate.
                pathway = "coherent"
                final_vertices_X = self.X[vertices_indices]
                final_weights = weights

                candidates_X = DirichletSampling(
                    concentration_factor=self.concentration_factor
                ).sample(
                    vertices_X=final_vertices_X,
                    weights=final_weights,
                    n_samples=n_samples,
                )

            else:
                # PATHWAY B1: Incoherent Triangle (TDA Hole Detected).
                # Fall back to bounded Gradient Descent using the single best anchor.
                pathway = "incoherent"

                best_idx_position = np.argmax(weights)
                single_best_vertex_idx = [vertices_indices[best_idx_position]]

                final_vertices_X = self.X[single_best_vertex_idx]
                final_weights = np.array([1.0])

                candidates_X = GradientDescentSampling(
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
            pathway = "extrapolation"

            nn_indices, nn_weights = self._get_nearest_neighbor(target_y)
            vertices_indices = (
                nn_indices  # Update metadata to reflect the single point used
            )

            final_vertices_X = self.X[nn_indices]
            final_weights = nn_weights

            candidates_X = GradientDescentSampling(
                forward_estimator=self.forward_estimator,
                target_y=target_y,
                trust_radius=self.trust_radius,
            ).sample(
                vertices_X=final_vertices_X,
                weights=final_weights,
                n_samples=n_samples,
            )

        # --- Step 3: Predict candidates objectives ---
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

    def _build_mesh(self, y: np.ndarray):
        """Builds the 2D Delaunay triangulation mesh in Objective (Y) space."""
        self.mesh = Delaunay(y)

    def _build_y_knn(self, y: np.ndarray):
        """Builds the spatial index for fast out-of-bounds nearest neighbor lookups."""
        self.objective_knn = NearestNeighbors(n_neighbors=1)
        self.objective_knn.fit(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Executes Phase 0: System Initialization & Pre-Processing."""
        self.forward_estimator = self._train_forward_estimator(X, y)

        # Build the TDA Vietoris-Rips complex instead of just calculating a scalar tau
        self._build_topological_complex(X)

        self._build_mesh(y)
        self.X = X
        self._build_y_knn(y)

    def history(self) -> dict[str, Any]:
        """Returns the offline training artifacts."""
        return {
            "vr_epsilon": self.vr_epsilon,
            "hole_penalty_factor": self.hole_penalty_factor,
        }
