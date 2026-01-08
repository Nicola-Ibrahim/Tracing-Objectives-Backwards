import numpy as np

from ...domain.modeling.interfaces.base_estimator import BaseEstimator
from ...domain.modeling.interfaces.base_normalizer import BaseNormalizer


class DecisionSampler:
    """Samples candidate decisions from an inverse estimator."""

    def sample(
        self,
        estimator: BaseEstimator,
        target_objective_norm: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Returns normalized candidates (n_samples, x_dim).
        Prefer probabilistic sampling if available.
        """
        candidates_norm = estimator.sample(target_objective_norm, n_samples=n_samples)

        # Normalize shapes to 2D: (n_candidates, x_dim).
        if candidates_norm.ndim == 3:
            candidates_norm = candidates_norm.reshape(-1, candidates_norm.shape[-1])

        return candidates_norm

    def denormalize(
        self,
        candidates_norm: np.ndarray,
        normalizer: BaseNormalizer,
    ) -> np.ndarray:
        """Converts normalized decisions to raw (physical) space."""
        return normalizer.inverse_transform(candidates_norm)
