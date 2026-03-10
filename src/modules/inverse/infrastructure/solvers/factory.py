from typing import Any, Type

from ....shared.infrastructure.inspection import (
    build_constructor_kwargs,
    extract_constructor_schema,
)
from ...domain.enums.solver_type import InverseSolverRegistry
from ...domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
)
from .gbpi.gbpi_solver import GBPIInverseSolver
from .gbpi.hybrid_solver import HybridGBPIInverseSolver
from .gbpi.tda_gbpi_solver import TDAGBPIInverseSolver
from .prob.mdn_solver import MDNProbabilisticInverseSolver


class SolversFactory:
    _registry: dict[InverseSolverRegistry, Type[AbstractInverseMappingSolver]] = {
        InverseSolverRegistry.GBPI: GBPIInverseSolver,
        InverseSolverRegistry.TDA_GBPI: TDAGBPIInverseSolver,
        InverseSolverRegistry.MDN: MDNProbabilisticInverseSolver,
        InverseSolverRegistry.HYBRID_GBPI: HybridGBPIInverseSolver,
    }

    def create(
        self, solver_type: InverseSolverRegistry | str, config: dict
    ) -> AbstractInverseMappingSolver:
        """
        Creates and returns an instance of an InverseSolver based on the specified type.
        Validates and normalizes incoming params (supports flat namespaces for Pydantic args).
        """
        if isinstance(solver_type, str):
            try:
                solver_type = InverseSolverRegistry(solver_type)
            except ValueError:
                raise ValueError(f"Unknown solver type: {solver_type}")

        solver_class = self._registry.get(solver_type)
        if not solver_class:
            raise ValueError(f"Solver type {solver_type} not registered.")

        # Build constructor arguments using centralized logic
        final_kwargs = build_constructor_kwargs(solver_class, config)

        return solver_class(**final_kwargs)

    def get_solver_schemas(self) -> list[dict[str, Any]]:
        """
        Returns a list of solver schemas for discovery purposes.
        """
        schemas = []
        for solver_id, solver_class in self._registry.items():
            schemas.append(
                {
                    "type": solver_id.value,
                    "name": solver_class.__name__.replace(
                        "InverseSolver", " Solver"
                    ).replace("Probabilistic", "Probabilistic "),
                    "parameters": extract_constructor_schema(solver_class),
                }
            )

        return schemas

    def create_from_config(
        self, config: dict[str, Any]
    ) -> AbstractInverseMappingSolver:
        """Deprecated: Use create instead."""
        solver_type = config.get("solver_type")
        if not solver_type:
            raise ValueError("Missing 'solver_type' in config")

        params = {k: v for k, v in config.items() if k != "solver_type"}
        return self.create(solver_type, params)
