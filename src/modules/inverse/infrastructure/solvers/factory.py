import inspect
from typing import Any, Dict, List, Type

from ....shared.infrastructure.inspection import (
    inspect_parameter,
    is_pydantic_model,
    normalize_value,
)
from ...domain.enums.solver_type import InverseSolverRegistry
from ...domain.interfaces.base_inverse_mapping_solver import (
    AbstractInverseMappingSolver,
)
from .gbpi.gbpi_solver import GBPIInverseSolver
from .gbpi.tda_gbpi_solver import TDAGBPIInverseSolver
from .prob.mdn_solver import MDNProbabilisticInverseSolver


class SolversFactory:
    _registry: Dict[InverseSolverRegistry, Type[AbstractInverseMappingSolver]] = {
        InverseSolverRegistry.GBPI: GBPIInverseSolver,
        InverseSolverRegistry.TDA_GBPI: TDAGBPIInverseSolver,
        InverseSolverRegistry.MDN: MDNProbabilisticInverseSolver,
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

        # Process config to handle grouped parameters (namespaces)
        sig = inspect.signature(solver_class.__init__)
        final_kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # 1. Check if the parameter is a Pydantic model
            if is_pydantic_model(param.annotation):
                # If the parameter is a Pydantic model, try to instantiate it from the flat config
                # First check if it's already provided as a dict under the parameter name
                if name in config and isinstance(config[name], dict):
                    final_kwargs[name] = param.annotation(**config[name])
                else:
                    # Otherwise, extract fields from the top-level config
                    model_fields = param.annotation.model_fields.keys()
                    filtered_config = {
                        k: v for k, v in config.items() if k in model_fields
                    }
                    # If we found fields or if it's required, try to instantiate
                    if filtered_config or param.default is inspect.Parameter.empty:
                        final_kwargs[name] = param.annotation(**filtered_config)
            else:
                # 2. Simple parameter: normalize (Enums, basic types)
                if name in config:
                    final_kwargs[name] = normalize_value(config[name], param.annotation)
                elif param.default is not inspect.Parameter.empty:
                    final_kwargs[name] = param.default

        return solver_class(**final_kwargs)

    def get_solver_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of solver schemas for discovery purposes.
        """
        schemas = []
        for solver_id, solver_class in self._registry.items():
            sig = inspect.signature(solver_class.__init__)
            parameters = []

            for name, param in sig.parameters.items():
                if name == "self":
                    continue

                parameters.extend(
                    inspect_parameter(
                        name=name,
                        annotation=param.annotation,
                        default=param.default,
                    )
                )

            schemas.append(
                {
                    "id": solver_id.value,
                    "name": solver_class.__name__.replace(
                        "InverseSolver", " Solver"
                    ).replace("Probabilistic", "Probabilistic "),
                    "parameters": parameters,
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
