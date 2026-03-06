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
    to detect non-linear manifolds and topological holes.
    """

    def __init__(
        self,
        n_neighbors: int,
        trust_radius: float,
        concentration_factor: float,
        hole_penalty_factor: float = 1.5,  # New: Ratio to detect a topological hole
    ):
        self.n_neighbors = n_neighbors
        self.trust_radius = trust_radius
        self.concentration_factor = concentration_factor
        self.hole_penalty_factor = hole_penalty_factor

        self.X = None
        self.mesh = None
        self.objective_knn = None
        self.forward_estimator = None

        # TDA Specific State
        self.vr_epsilon = None  # The Vietoris-Rips linking radius
        self.vr_graph = None  # The structural 1-skeleton of the dataset

    def type(self):
        return "TDA-GBPI"

    def _ensure_fitted(self):
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
        This maps the "shape" of the data and finds connected components.
        """
        # 1. Find the characteristic local scale (epsilon)
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        self.vr_epsilon = float(np.percentile(distances[:, 1:], 95))

        # 2. Build the adjacency matrix (connect points if distance <= epsilon)
        dist_mat = distance_matrix(X, X)
        # Create a boolean mask of valid connections, then multiply by actual distances to weight the graph
        adjacency = np.where(dist_mat <= self.vr_epsilon, dist_mat, 0)

        # 3. Store as a sparse graph for fast topological traversal
        self.vr_graph = csr_matrix(adjacency)

    def _evaluate_coherence(
        self, vertices_indices: list[int]
    ) -> tuple[bool, list[float]]:
        """
        Enforces TDA Coherence. Checks if the straight Euclidean line cuts across a
        topological hole by comparing it to the geodesic path on the Vietoris-Rips graph.
        """
        triangle_vertices = self.X[vertices_indices]

        if len(triangle_vertices) < 2:
            return True, []

        is_coherent = True
        anchor_distances = []

        # Check all pairs of the triangle (A-B, B-C, A-C)
        for i in range(len(vertices_indices)):
            for j in range(i + 1, len(vertices_indices)):
                idx_A = vertices_indices[i]
                idx_B = vertices_indices[j]

                # 1. Calculate raw Euclidean distance (the straight line)
                euclidean_dist = np.linalg.norm(self.X[idx_A] - self.X[idx_B])
                anchor_distances.append(euclidean_dist)

                # 2. Calculate the Topological Geodesic distance (the structural path)
                # This finds the shortest path safely traversing the data manifold
                geodesic_dist = dijkstra(
                    csgraph=self.vr_graph,
                    directed=False,
                    indices=idx_A,
                    return_predecessors=False,
                )[idx_B]

                # 3. The TDA Hole Detection Logic
                # If geodesic is infinite, they are in completely disconnected clusters.
                if np.isinf(geodesic_dist):
                    is_coherent = False
                    break

                # If the path along the manifold is significantly longer than the straight line,
                # it means the straight line cuts across a topological hole (like a horseshoe void).
                if geodesic_dist > (euclidean_dist * self.hole_penalty_factor):
                    is_coherent = False
                    break

        return is_coherent, anchor_distances

    def _locate(self, target: np.ndarray) -> tuple[list[int], np.ndarray, bool, bool]:
        """
        Locates the target within the Delaunay mesh or its nearest neighbor.
        (Remains identical to original GBPI implementation)
        """
        target = np.asarray(target).reshape(1, -1)
        if target.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        simplex_idx = self.mesh.find_simplex(target)[0]

        if simplex_idx != -1:
            transform = self.mesh.transform[simplex_idx]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]
            vertices_indices = self.mesh.simplices[simplex_idx].tolist()
            is_simplex_found = True

            is_coherent, anchor_distances = self._evaluate_coherence(vertices_indices)
            return (
                vertices_indices,
                weights,
                is_simplex_found,
                is_coherent,
                anchor_distances,
            )

        else:
            _, indices = self.objective_knn.kneighbors(target)
            closest_vertex_idx = int(indices[0][0])
            vertices_indices = [closest_vertex_idx]
            weights = np.array([1.0])
            is_simplex_found = False
            is_coherent = False
            anchor_distances = []
            return (
                vertices_indices,
                weights,
                is_simplex_found,
                is_coherent,
                anchor_distances,
            )

    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        """
        (Remains identical to original GBPI implementation)
        """
        self._ensure_fitted()

        (
            vertices_indices,
            weights,
            is_simplex_found,
            is_coherent,
            anchor_distances,
        ) = self._locate(target_y)

        if is_coherent and is_simplex_found:
            pathway = "coherent"
            candidates_X = DirichletSampling(
                concentration_factor=self.concentration_factor
            ).sample(
                vertices=self.X[vertices_indices],
                weights=weights,
                n_samples=n_samples,
            )
        else:
            pathway = "incoherent"
            candidates_X = GradientDescentSampling(
                forward_estimator=self.forward_estimator,
                target_y=target_y,
                trust_radius=self.trust_radius,
            ).sample(
                vertices=self.X[vertices_indices],
                weights=weights,
                n_samples=n_samples,
            )

        candidates_y = self.forward_estimator.predict(candidates_X)

        return InverseSolverResult(
            candidates_X=candidates_X,
            candidates_y=candidates_y,
            metadata={
                "pathway": pathway,
                "is_simplex_found": is_simplex_found,
                "is_coherent": is_coherent,
                "anchor_distances": anchor_distances,
                "vertices_indices": vertices_indices,
            },
        )

    def _train_forward_estimator(self, X: np.ndarray, y: np.ndarray):
        params = RBFEstimatorParams(n_neighbors=10)
        estimator = RBFEstimator(params)
        estimator.fit(X, y)
        return estimator

    def _build_mesh(self, y: np.ndarray):
        self.mesh = Delaunay(y)

    def _build_y_knn(self, y: np.ndarray):
        self.objective_knn = NearestNeighbors(n_neighbors=1)
        self.objective_knn.fit(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.forward_estimator = self._train_forward_estimator(X, y)

        # NEW: Build the TDA Vietoris-Rips complex instead of just calculating tau
        self._build_topological_complex(X)

        self._build_mesh(y)
        self.X = X
        self._build_y_knn(y)

    def history(self) -> dict[str, Any]:
        return {
            "vr_epsilon": self.vr_epsilon,
            "hole_penalty_factor": self.hole_penalty_factor,
        }
