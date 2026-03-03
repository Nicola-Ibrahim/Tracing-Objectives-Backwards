from typing import Any

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
        if solver_type in self._registry:
            return self._registry[solver_type](**kwargs)
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
