import cocoex
import numpy as np
from cocoex import Problem as COCOProblem

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
        # Persist configuration so the estimator can be reconstructed after serialization
        self.problem_name = problem_name
        self.function_indices = function_indices
        self.instance_indices = instance_indices
        self.dimensions = dimensions
        self._problem = self._build_problem(
            problem_name,
            function_indices,
            instance_indices,
            dimensions,
        )

    @staticmethod
    def _build_problem(
        problem_name: str,
        function_indices: int,
        instance_indices: int,
        dimensions: int,
    ) -> COCOProblem:
        """Initialize a COCO BBOB-BIOBJ problem with specified configuration."""

        if problem_name == "bbob-biobj" and not (1 <= function_indices <= 55):
            raise ValueError(
                "`function_indices` must be between 1 and 55 for suite "
                f"'{problem_name}', got {function_indices}."
            )

        suite_options = (
            f"dimensions:{dimensions} "
            f"instance_indices:{instance_indices} "
            f"function_indices:{function_indices}"
        )

        suite = cocoex.Suite(
            problem_name,
            "",
            suite_options,
        )

        return suite.get_problem(0)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # COCO problems cannot be pickled; rebuild on load instead.
        state.pop("_problem", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._problem = self._build_problem(
            self.problem_name,
            self.function_indices,
            self.instance_indices,
            self.dimensions,
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
