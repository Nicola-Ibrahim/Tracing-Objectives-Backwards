from typing import Any

from ....modeling.infrastructure.estimators.mdn_estimator import MDNEstimator
from ...domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
)
from .gbpi.solver import GBPIInverseSolver
from .prob.solver import ProbabilisticInverseSolver


class SolversFactory:
    _registry = {
        "GBPI": GBPIInverseSolver,
        "probabilistic": ProbabilisticInverseSolver,
    }

    def create(self, solver_type: str, **kwargs) -> AbstractInverseMappingSolver:
        if solver_type == "MDN":
            estimator = MDNEstimator(**kwargs)
            return ProbabilisticInverseSolver(estimator=estimator)

        if solver_type in self._registry:
            # Check if it requires an estimator or can be created with kwargs
            solver_class = self._registry[solver_type]
            if solver_class == ProbabilisticInverseSolver:
                # If "probabilistic" is requested directly, it needs an estimator in kwargs
                # or we just let it fail if not provided.
                return solver_class(**kwargs)
            return solver_class(**kwargs)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    def create_from_config(
        self, config: dict[str, Any]
    ) -> AbstractInverseMappingSolver:
        solver_type = config.get("solver_type")
        if solver_type in self._registry:
            return self._registry[solver_type](**config)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
