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
from .sampling.dirichlet import DirichletSampling
from .sampling.gd import GradientDescentSampling


class GBPIInverseSolver(AbstractInverseMappingSolver):
    """
    Concrete implementation of the GBPI framework.
    It explicitly owns its specific mathematical state.
    """

    def __init__(
        self,
        n_neighbors: int,
        trust_radius: float,
        concentration_factor: float,
        estimator: Any = None,
    ):
        """
        Initializes the GBPISolver.

        Args:
            n_neighbors: The number of neighbors to use for the KNN.
            trust_radius: The trust-region radius for the gradient descent.
            concentration_factor: The concentration factor for the Dirichlet distribution.
            estimator: The estimator to use for the forward mapping.
        """
        self.n_neighbors = n_neighbors
        self.trust_radius = trust_radius
        self.concentration_factor = concentration_factor
        self.estimator = estimator

        self.X = None
        self.tau = None
        self.mesh = None
        self.objective_knn = None
        self.forward_estimator = None

    def type(self):
        return "GBPI"

    def _ensure_fitted(self):
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
        """Enforces the coherence domain rule based on the tau threshold
        Checks if all pairwise distances between vertices are less than or equal to tau.
        Returns:
            is_coherent: True if all pairwise distances <= tau.
            pairwise_dists: List of pairwise Euclidean distances.
        """
        triangle_vertices = self.X[vertices_indices]

        if len(triangle_vertices) < 2:
            return True, []

        # Calculate pairwise distances between vertices
        diffs = triangle_vertices[:, None, :] - triangle_vertices[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        i, j = np.triu_indices(len(triangle_vertices), k=1)
        pairwise_dists = dists[i, j].tolist()
        is_coherent = bool(np.all(np.array(pairwise_dists) <= self.tau))
        return is_coherent, pairwise_dists

    def _locate(self, target: np.ndarray) -> tuple[list[int], np.ndarray, bool, bool]:
        """
        Locates the target within the Delaunay mesh or its nearest neighbor.

        Uses the Delaunay triangulation if the target is within the convex hull.
        Falls back to an optimized KNN lookup (KDTree) if the target is outside.

        Args:
            target: (1, 2) or (2,) array representing the target objective.

        Returns:
            anchor_indices: List of original indices forming the local geometry or nearest point.
            weights: Barycentric weights for Delaunay, or [1.0] for KNN.
            is_simplex_found: True if the target is inside a Delaunay simplex.
            is_coherent: True if the local geometry is coherent (not stretched).
            anchor_distances: List of pairwise distances between anchors.
        """
        target = np.asarray(target).reshape(1, -1)
        if target.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        # find the single triangle that the target landed inside
        simplex_idx = self.mesh.find_simplex(target)[0]

        # if target is inside the mesh, use Delaunay triangulation
        # and return the vertices indices and the barycentric weights
        # if simplex index is positive, means a Triangle was found
        if simplex_idx != -1:
            transform = self.mesh.transform[simplex_idx]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]  # barycentric weights
            vertices_indices = self.mesh.simplices[
                simplex_idx
            ].tolist()  # indices of the vertices (corners) forming the triangle
            is_simplex_found = True

            # check if the triangle is coherent (not dangerously stretched out)
            is_coherent, anchor_distances = self._evaluate_coherence(vertices_indices)
            return (
                vertices_indices,
                weights,
                is_simplex_found,
                is_coherent,
                anchor_distances,
            )

        # if target is outside the mesh, find the nearest point in y
        # and return the nearest point index and the barycentric weights
        # if simplex index is negative, means no triangle was found
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
        # Check if it trained or not
        self._ensure_fitted()

        # Check if target is inside or outside the mesh
        (
            vertices_indices,
            weights,
            is_simplex_found,
            is_coherent,
            anchor_distances,
        ) = self._locate(target_y)

        # 2. Sample candidate decisions by strategy
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

        # Predict candidates objectives
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
        """
        Trains the forward estimator.
        Args:
            X: (N, D) array representing the design space points
            y: (N, D) array representing the objective space points
        """
        params = RBFEstimatorParams(n_neighbors=10)
        estimator = RBFEstimator(params)
        estimator.fit(X, y)
        return estimator

    def _calculate_coherence_threshold(self, X: np.ndarray):
        """
        Calculates the coherence threshold (tau) based on the 95th percentile of the nearest neighbor distances.
        Args:
            X: (N, D) array representing the design space points
        """
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)  # +1 to include self
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # distances[:, 1:] ignores the zero-distance to self
        self.tau = float(np.percentile(distances[:, 1:], 95))

    def _build_mesh(self, y: np.ndarray):
        """
        Builds the Delaunay triangulation of the objective space.
        It is used to find the simplex that the target point is inside.
        Args:
            y: (N, D) array representing the objective space points
        """
        self.mesh = Delaunay(y)

    def _build_y_knn(self, y: np.ndarray):
        """
        Builds the spatial index for the objective space (for out-of-mesh lookups)
        It is used to find the nearest neighbor of a target point that is outside the mesh.
        Args:
            y: (N, D) array representing the objective space points
        """
        self.objective_knn = NearestNeighbors(n_neighbors=1)
        self.objective_knn.fit(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the GBPI solver.
        Args:
            X: (N, D) array representing the design space points
            y: (N, D) array representing the objective space points
        """
        # Training the Surrogate Model (The actual X -> Y mapping)
        self.forward_estimator = self._train_forward_estimator(X, y)

        # Computing the Coherence Threshold (tau)
        self._calculate_coherence_threshold(X)

        # Create the mesh for the objective space
        self._build_mesh(y)

        # Store the raw training data
        self.X = X

        # Build the spatial index for the objective space (for out-of-mesh lookups)
        self._build_y_knn(y)

    def history(self) -> dict[str, Any]:
        """Returns the history of the solver."""
        return {
            "tau": self.tau,
        }
