import numpy as np
from scipy.spatial import Delaunay


class BarycentricLocator:
    """
    Domain service for locating a target objective within the Delaunay mesh of known objectives.
    """

    @staticmethod
    def locate(
        target: np.ndarray, objectives: np.ndarray
    ) -> tuple[list[int], np.ndarray, bool]:
        """
        Locates the target within the Delaunay mesh.

        Args:
            target: (1, 2) or (2,) array representing the target objective.
            objectives: (N, 2) array of known objectives forming the mesh.

        Returns:
            anchor_indices: List of original indices (in `objectives`) forming the local geometry.
            weights: Barycentric weights corresponding to the anchors.
            is_inside: True if the target falls inside the convex hull.
        """
        target = np.asarray(target).reshape(1, -1)
        if target.shape[1] != 2:
            raise ValueError("Target must be a 2D coordinate")

        tri = Delaunay(objectives)
        simplex_index = tri.find_simplex(target)[0]

        if simplex_index != -1:
            # Inside the mesh
            transform = tri.transform[simplex_index]
            inv_T = transform[:2, :2]
            r = transform[2, :]

            b = inv_T.dot(target[0] - r)
            weights = np.r_[b, 1.0 - b.sum()]

            anchor_indices = tri.simplices[simplex_index].tolist()
            return anchor_indices, weights, True
        else:
            # Fallback for target outside mesh (nearest point)
            dists = np.linalg.norm(objectives - target, axis=1)
            closest_idx = int(np.argmin(dists))
            return [closest_idx], np.array([1.0]), False
