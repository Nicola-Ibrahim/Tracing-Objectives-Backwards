import numpy as np

from .....modeling.domain.interfaces.base_estimator import (
    ProbabilisticEstimator,
)
from ....domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
    InverseSolverResult,
)


class ProbabilisticInverseSolver(AbstractInverseMappingSolver):
    """
    Adapter that executes inverse mapping by sampling from a probabilistic estimator.
    """

    def __init__(self, estimator: ProbabilisticEstimator):
        self.estimator = estimator

    def type(self) -> str:
        return "probabilistic"

    def _ensure_fitted(self) -> None:
        self.estimator._ensure_fitted()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the internal probabilistic estimator on the provided data.
        """
        self.estimator.fit(y, X)

    def generate(self, target_y: np.ndarray, n_samples: int) -> InverseSolverResult:
        """
        Samples candidate decision vectors from the inverse distribution learned by the estimator.
        """
        self._ensure_fitted()

        # target_y shape is expected to be (n_objectives,) or (1, n_objectives)
        # We ensure it's (1, n_objectives) for sampling if needed, or follow the estimator API
        # MDN/CVAE/INN usually expect a batch dimension.
        if target_y.ndim == 1:
            query_y = target_y.reshape(1, -1)
        else:
            query_y = target_y

        candidates_x = self.estimator.sample(query_y, n_samples=n_samples)

        # candidates_x shape: (n_samples, n_decisions)
        # We also need candidates_y (the objectives of these candidates).
        # For a probabilistic solver, the candidates_y are conceptually 'target_y'
        # but to be rigorous we might want to predict them if it's a surrogate,
        # however since it's an inverse model, target_y IS the objective we are aiming for.
        # We tile target_y to match candidates_x.
        candidates_y = np.tile(target_y, (n_samples, 1))

        return InverseSolverResult(
            candidates_X=candidates_x, candidates_y=candidates_y, metadata={}
        )
