from typing import Any

import numpy as np
from scipy.optimize import minimize
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


class HybridGBPIInverseSolver(AbstractInverseMappingSolver):
    """
    State-of-the-art Hybrid implementation of the GBPI framework.
    Uses Dirichlet sampling to find a diverse, physically safe neighborhood,
    and then applies a Gradient Descent 'polish' to eliminate non-linear
    Pareto Sag and pull candidates exactly onto the user target.
    """

    def __init__(
        self,
        coherence_n_neighbors: int = 5,
        trust_radius: float = 0.05,
        concentration_factor: float = 50.0,
    ):
        self.coherence_n_neighbors = coherence_n_neighbors
        self.trust_radius = trust_radius
        self.concentration_factor = concentration_factor

        self.X = None
        self.tau = None
        self.mesh = None
        self.objective_knn = None
        self.forward_estimator = None

    def type(self):
        return "Hybrid-GBPI"

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
        simplex_idx = self.mesh.find_simplex(target)[0]

        if simplex_idx != -1:
            transform = self.mesh.transform[simplex_idx]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]
            vertices_indices = self.mesh.simplices[simplex_idx].tolist()
            return True, vertices_indices, weights
        else:
            return False, [], np.array([])

    def _get_nearest_neighbor(self, target: np.ndarray) -> tuple[list[int], np.ndarray]:
        _, indices = self.objective_knn.kneighbors(target)
        closest_vertex_idx = int(indices[0][0])
        return [closest_vertex_idx], np.array([1.0])

    def _polish_candidates(
        self, raw_candidates_X: np.ndarray, target_y: np.ndarray
    ) -> np.ndarray:
        """
        The Core Hybrid Feature: Takes the safe Dirichlet candidates and uses L-BFGS-B
        optimization to gently drag them exactly to the target_y, eliminating drift.
        It uses the trust_radius to ensure it doesn't break physical bounds.
        """
        polished_X = []
        target_y_flat = target_y.flatten()

        # Define the objective function (Minimize the squared error in Y-space)
        def objective_func(x):
            # predict expects a 2D array, so we reshape x
            pred_y = self.forward_estimator.predict(x.reshape(1, -1)).flatten()
            return np.sum((pred_y - target_y_flat) ** 2)

        for x0 in raw_candidates_X:
            # Set strict physical boundaries around the Dirichlet guess
            # This ensures the polish only fixes the sag, it doesn't hallucinate.
            bounds = [
                (max(0.0, val - self.trust_radius), min(1.0, val + self.trust_radius))
                for val in x0
            ]

            # Run the optimizer
            res = minimize(
                objective_func,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 20},  # Keep it fast for real-time latency
            )
            polished_X.append(res.x)

        return np.array(polished_X)

    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        self._ensure_fitted()

        target_y = np.asarray(target_y).reshape(1, -1)
        if target_y.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        is_inside_mesh, vertices_indices, weights = self._locate_in_mesh(target_y)

        is_coherent = False
        anchor_distances = []
        pathway = "unknown"

        if is_inside_mesh:
            is_coherent, anchor_distances = self._evaluate_coherence(vertices_indices)

            if is_coherent:
                # --- THE HYBRID UPGRADE ---
                pathway = "coherent_polished"

                # Step 1: Get diverse, safe candidates from Dirichlet
                raw_candidates_X = CoherentSampling(
                    concentration_factor=self.concentration_factor
                ).sample(
                    vertices_X=self.X[vertices_indices],
                    weights=weights,
                    n_samples=n_samples,
                )

                # Step 2: Polish them to snap directly onto the green cross target
                candidates_X = self._polish_candidates(raw_candidates_X, target_y)

            else:
                pathway = "incoherent"
                best_idx_position = np.argmax(weights)
                single_best_vertex_idx = [vertices_indices[best_idx_position]]

                candidates_X = IncoherentSampling(
                    forward_estimator=self.forward_estimator,
                    target_y=target_y,
                    trust_radius=self.trust_radius,
                ).sample(
                    vertices_X=self.X[single_best_vertex_idx],
                    weights=np.array([1.0]),
                    n_samples=n_samples,
                )
        else:
            pathway = "extrapolation"
            nn_indices, nn_weights = self._get_nearest_neighbor(target_y)
            vertices_indices = nn_indices

            candidates_X = ExtrapolationSampling(
                forward_estimator=self.forward_estimator,
                target_y=target_y,
                trust_radius=self.trust_radius,
            ).sample(
                vertices_X=self.X[nn_indices],
                weights=nn_weights,
                n_samples=n_samples,
            )

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
        params = RBFEstimatorParams(n_neighbors=10)
        estimator = RBFEstimator(params)
        estimator.fit(X, y)
        return estimator

    def _calculate_coherence_threshold(self, X: np.ndarray):
        nn = NearestNeighbors(n_neighbors=self.coherence_n_neighbors + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        self.tau = float(np.percentile(distances[:, 1:], 95))

    def _build_mesh(self, y: np.ndarray):
        self.mesh = Delaunay(y)

    def _build_y_knn(self, y: np.ndarray):
        self.objective_knn = NearestNeighbors(n_neighbors=1)
        self.objective_knn.fit(y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.forward_estimator = self._train_forward_estimator(X, y)
        self._calculate_coherence_threshold(X)
        self._build_mesh(y)
        self.X = X
        self._build_y_knn(y)

    def history(self) -> dict[str, Any]:
        return {"tau": self.tau}
