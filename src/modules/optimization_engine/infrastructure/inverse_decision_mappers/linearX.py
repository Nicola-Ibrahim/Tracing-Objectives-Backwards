from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ...utils.preprocessing.similarities import SimilarityMethod
from ...modules.interpolation.domain.models.preference import ObjectivePreferences
from ...domain.interfaces.base_interpolator import BaseInterpolator


class LocalXInterpolator(BaseInterpolator):
    """
    Base class for local interpolators in X-space that find decision vectors
    to match desired objectives via inverse design.
    """

    def __init__(
        self, similarity_metric: SimilarityMethod, k_neighbors: int = 5
    ) -> None:
        """
        Initialize the Local Interpolator.

        Args:
            similarity_metric (SimilarityMethod): Metric used to compare objective vectors.
            k_neighbors (int): Number of nearest neighbors to use for local interpolation
        """
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self._all_decision_vectors: Optional[NDArray[np.float64]] = None
        self._all_objective_vectors: Optional[NDArray[np.float64]] = None
        self._initial_sort_indices: Optional[NDArray[np.int_]] = None
        self.forward_model: Optional[callable] = None
        self.X_mean: Optional[NDArray[np.float64]] = None
        self.X_std: Optional[NDArray[np.float64]] = None
        self.Y_mean: Optional[NDArray[np.float64]] = None
        self.Y_std: Optional[NDArray[np.float64]] = None

    def fit(
        self,
        candidate_solutions: NDArray[np.float64],
        objective_front: NDArray[np.float64],
    ) -> None:
        """
        Fits the interpolator with the entire set of candidate solutions and objective vectors.
        Also builds a global forward model for objective prediction.

        Args:
            candidate_solutions (NDArray[np.float64]): Array of all candidate solutions in decision space.
            objective_front (NDArray[np.float64]): Array of all corresponding objective vectors.

        Raises:
            ValueError: If lengths of inputs do not match or are empty.
        """
        if candidate_solutions.shape[0] != objective_front.shape[0]:
            raise ValueError(
                "Candidate solutions and objective front must be of the same length."
            )
        if candidate_solutions.shape[0] == 0:
            raise ValueError("Input data cannot be empty for fitting the interpolator.")

        # Store the original sort order to maintain mapping from input indices
        self._initial_sort_indices = np.argsort(objective_front[:, 0])
        self._all_decision_vectors = candidate_solutions[
            self._initial_sort_indices
        ].astype(np.float64)
        self._all_objective_vectors = objective_front[
            self._initial_sort_indices
        ].astype(np.float64)

        # Build global forward model (to be implemented in subclasses)
        self._build_forward_model(X_norm, Y_norm)

    def _build_forward_model(
        self, X_norm: NDArray[np.float64], Y_norm: NDArray[np.float64]
    ) -> None:
        """
        Build forward model in normalized space. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_active_region_data(
        self, active_region_indices: Optional[Sequence[int]] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Retrieve data for the active region, handling mapping between original and sorted indices.

        Args:
            active_region_indices: Indices of solutions defining the active region

        Returns:
            Tuple of decision vectors and objective vectors for the active region
        """
        if active_region_indices is None:
            return self._all_decision_vectors, self._all_objective_vectors

        # Map active_region_indices (original indices) to sorted indices
        inv_perm = np.argsort(self._initial_sort_indices)
        sorted_indices_of_active = inv_perm[active_region_indices]

        return (
            self._all_decision_vectors[sorted_indices_of_active],
            self._all_objective_vectors[sorted_indices_of_active],
        )

    def _find_neighbors(
        self,
        target: NDArray[np.float64],
        objectives: NDArray[np.float64],
        decisions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Find k nearest neighbors to target in objective space.

        Args:
            target: Target objective vector
            objectives: All objective vectors in active region
            decisions: Corresponding decision vectors

        Returns:
            Tuple of neighbor decision vectors and neighbor objective vectors
        """
        # Normalize for distance calculation
        objectives_norm = (objectives - self.Y_mean) / self.Y_std
        target_norm = (target - self.Y_mean) / self.Y_std

        # Calculate distances
        distances = np.linalg.norm(objectives_norm - target_norm, axis=1)

        # Find k nearest neighbors
        k = min(self.k_neighbors, len(objectives))
        neighbor_indices = np.argpartition(distances, k)[:k]

        return decisions[neighbor_indices], objectives[neighbor_indices]

    def generate(
        self,
        user_preferences: ObjectivePreferences,
        active_region_indices: Optional[Sequence[int]] = None,
    ) -> NDArray[np.float64]:
        """
        Generates a recommended decision vector by:
        1. Finding neighbors near target in objective space
        2. Building a local interpolator in decision space
        3. Optimizing to minimize objective-space error

        Args:
            user_preferences: User's preferences defining the target
            active_region_indices: Indices defining the active region

        Returns:
            Optimal decision vector
        """
        # Validate state
        if None in (
            self._all_decision_vectors,
            self._all_objective_vectors,
            self.forward_model,
        ):
            raise RuntimeError("Interpolator has not been properly fitted.")

        # Get target and active region data
        target = np.array(user_preferences.weights, dtype=np.float64)
        decisions, objectives = self._get_active_region_data(active_region_indices)

        if len(decisions) == 0:
            raise ValueError("No data points in the active region.")

        # Find neighbors near target
        X_neighbors, Y_neighbors = self._find_neighbors(target, objectives, decisions)

        # Handle case with only one neighbor
        if len(X_neighbors) == 1:
            return X_neighbors[0]

        # Build local interpolator in X-space
        local_interpolator = self._build_local_interpolator(X_neighbors, Y_neighbors)

        # Optimize to find X that minimizes ||f(X) - target||²
        return self._optimize(local_interpolator, X_neighbors, target)

    def _build_local_interpolator(
        self,
        X_neighbors: NDArray[np.float64],
        Y_neighbors: NDArray[np.float64],
    ) -> callable:
        """
        Build local interpolator in X-space. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _optimize(
        self,
        local_interpolator: callable,
        X_neighbors: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Optimize to find X that minimizes the objective-space error.
        """

        # Define loss function
        def loss(X: NDArray[np.float64]) -> float:
            X_norm = (X - self.X_mean) / self.X_std
            Y_pred_norm = self.forward_model(X_norm.reshape(1, -1))[0]
            Y_pred = Y_pred_norm * self.Y_std + self.Y_mean
            return np.sum((Y_pred - target) ** 2)

        # Set up optimization
        x0 = np.mean(X_neighbors, axis=0)  # Start from centroid
        bounds = self._get_bounds(X_neighbors)

        result = minimize(
            loss,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        if not result.success:
            # Fallback: return the neighbor closest to target
            distances = np.linalg.norm(X_neighbors - result.x, axis=1)
            return X_neighbors[np.argmin(distances)]

        return result.x

    def _get_bounds(
        self, X_neighbors: NDArray[np.float64]
    ) -> list[tuple[float, float]]:
        """
        Get bounds for optimization based on neighbors.
        """
        mins = np.min(X_neighbors, axis=0)
        maxs = np.max(X_neighbors, axis=0)
        return [(mins[i], maxs[i]) for i in range(X_neighbors.shape[1])]
