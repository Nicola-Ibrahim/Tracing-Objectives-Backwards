import numpy as np

from .....domain.modeling.interfaces.base_estimator import DeterministicEstimator


class COCOEstimator(DeterministicEstimator):
    """
    Deterministic forward estimator backed by COCO bi-objective problems.
    Maps decisions X -> objectives F using the configured COCO problem.
    """

    def __init__(
        self,
        *,
        problem_name: str = "bbob-biobj",
        function_indices: int = 5,
        instance_indices: int = 1,
        dimensions: int = 2,
    ) -> None:
        super().__init__(
            problem_name=problem_name,
            function_indices=function_indices,
            instance_indices=instance_indices,
            dimensions=dimensions,
        )
        self._problem = get_coco_problem(
            problem_name=problem_name,
            function_indices=function_indices,
            instance_indices=instance_indices,
            dimensions=dimensions,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # COCO problem returns objective vector; ensure 2D output
        F = np.array([self._problem(x) for x in X], dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        return F
