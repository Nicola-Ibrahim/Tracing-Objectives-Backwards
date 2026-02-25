import numpy as np


class CoherenceGate:
    """
    Domain service acting as a safeguard to prevent generation of physically impossible configurations.
    """

    @staticmethod
    def check(anchors_norm: np.ndarray, tau: float) -> bool:
        """
        Checks if the bounding anchors represent a mathematically coherent local region.

        Args:
            anchors_norm: (N, D) array of normalized decision configurations for the anchors.
            tau: The absolute coherence threshold.

        Returns:
            True if coherent (all pairwise distances <= tau), False otherwise.
        """
        if len(anchors_norm) < 2:
            return True

        # Calculate pairwise euclidean distances
        diffs = anchors_norm[:, None, :] - anchors_norm[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Get upper triangle, excluding diagonal
        i, j = np.triu_indices(len(anchors_norm), k=1)
        pairwise_dists = dists[i, j]

        return bool(np.all(pairwise_dists <= tau))
