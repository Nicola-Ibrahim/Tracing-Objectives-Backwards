import cocoex
import numpy as np
from cocoex import Problem as COCOProblem

from .....domain.modeling.enums.estimator_type import EstimatorTypeEnum
from .....domain.modeling.interfaces.base_estimator import DeterministicEstimator
from .....domain.modeling.value_objects.estimator_params import COCOEstimatorParams


class COCOEstimator(DeterministicEstimator):
    """
    Deterministic forward estimator backed by COCO bi-objective problems.
    Maps decisions X -> objectives F using the configured COCO problem.
    """

    def __init__(self, params: COCOEstimatorParams) -> None:
        super().__init__()
        self.params = params
        self.problem_name = params.problem_name
        self.function_indices = params.function_indices
        self.instance_indices = params.instance_indices
        self.dimensions = params.dimensions

        self._problem = self._build_problem(
            self.problem_name,
            self.function_indices,
            self.instance_indices,
            self.dimensions,
        )

    @property
    def type(self) -> str:
        """
        Returns the type of the inverse decision mapper.
        This should be overridden by subclasses to return their specific type.
        """
        return getattr(EstimatorTypeEnum, "COCO", EstimatorTypeEnum.COCO).value

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

    # -------------------- Checkpoint Serialization --------------------

    def to_checkpoint(self) -> dict:
        """
        Serialize COCO estimator configuration to JSON-serializable checkpoint.

        COCO problems are deterministic and don't have trainable parameters,
        so we only need to store the configuration.

        Returns:
            dict: Checkpoint containing problem configuration
        """
        checkpoint = {
            "problem_name": self.problem_name,
            "function_indices": int(self.function_indices),
            "instance_indices": int(self.instance_indices),
            "dimensions": int(self.dimensions),
        }
        return checkpoint

    @classmethod
    def from_checkpoint(cls, parameters: dict) -> "COCOEstimator":
        """
        Reconstruct COCO estimator from parameters.

        Args:
            parameters: Full parameters dict containing configuration

        Returns:
            COCOEstimator: Fully initialized estimator
        """
        # For COCO, parameters contain the configuration
        # Filter out metadata fields
        config = {
            k: v
            for k, v in parameters.items()
            if k
            in ["problem_name", "function_indices", "instance_indices", "dimensions"]
        }

        return cls(params=COCOEstimatorParams(**config))
