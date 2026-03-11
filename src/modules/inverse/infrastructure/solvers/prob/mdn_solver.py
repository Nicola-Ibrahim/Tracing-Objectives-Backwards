from typing import Any

import numpy as np

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum
from .....modeling.infrastructure.estimators.deterministic.rbf import (
    RBFEstimator,
    RBFEstimatorParams,
)
from .....modeling.infrastructure.estimators.probabilistic.mdn import MDNEstimatorParams
from .....modeling.infrastructure.factories.estimator import (
    EstimatorFactory,
)
from ....domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
    InverseSolverResult,
)


class MDNProbabilisticInverseSolver(AbstractInverseMappingSolver):
    """
    Adapter that executes inverse mapping by sampling from a probabilistic estimator.
    """

    def __init__(self, params: MDNEstimatorParams):
        self.estimator = EstimatorFactory().create(
            type=EstimatorTypeEnum.MDN, params=params
        )
        self.forward_estimator: RBFEstimator | None = None

    def type(self) -> str:
        return "MDN-Probabilistic"

    def _ensure_fitted(self) -> None:
        self.estimator._ensure_fitted()
        if self.forward_estimator is None:
            raise RuntimeError("Forward estimator not fitted. Call 'train' first.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the internal probabilistic estimator and forward RBF estimator.
        """
        # Inverse mapping: y -> X
        self.estimator.fit(y, X)

        # Forward mapping for candidate evaluation: X -> y
        rbf_params = RBFEstimatorParams(n_neighbors=7, kernel="thin_plate_spline")
        self.forward_estimator = RBFEstimator(rbf_params)
        self.forward_estimator.fit(X, y)

    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        """
        Samples candidate decision vectors from the inverse distribution learned by the estimator.
        """
        self._ensure_fitted()

        # target_y shape is expected to be (n_objectives,) or (1, n_objectives)
        if target_y.ndim == 1:
            query_y = target_y.reshape(1, -1)
        else:
            query_y = target_y

        # estimator.sample returns (N, n_samples, output_dim)
        # For N=1, we want (n_samples, output_dim)
        candidates_raw = self.estimator.sample(query_y, n_samples=n_samples)
        candidates_x = candidates_raw.reshape(-1, candidates_raw.shape[-1])

        # Use forward estimator to predict objective values for candidates
        candidates_y = self.forward_estimator.predict(candidates_x)

        metadata = {}
        if hasattr(self.estimator, "get_log_likelihood"):
            # Compute log-likelihood for each candidate
            repeated_y = np.repeat(query_y, n_samples, axis=0)
            ll = self.estimator.get_log_likelihood(repeated_y, candidates_x)
            metadata["log_likelihood"] = ll.tolist()

        return InverseSolverResult(
            candidates_X=candidates_x, candidates_y=candidates_y, metadata=metadata
        )

    def history(self) -> dict[str, Any]:
        """Returns the history of the solver."""
        return {
            "loss": self.estimator.get_loss_history(),
        }
