import inspect
from typing import Any, Dict, List, Type

from ....shared.infrastructure.inspection import (
    inspect_parameter,
    is_pydantic_model,
    normalize_value,
)
from ...domain.enums.generator_type import DatasetGeneratorRegistry
from ...domain.interfaces.base_data_source import BaseDataSource
from .pymoo.generate import PymooGenerator


class DataGeneratorFactory:
    _registry: Dict[DatasetGeneratorRegistry, Type[BaseDataSource]] = {
        DatasetGeneratorRegistry.COCO_PYMOO: PymooGenerator,
    }

    def create(
        self, generator_type: DatasetGeneratorRegistry | str, params: dict[str, Any]
    ) -> BaseDataSource:
        """
        Creates a data source generator based on the type and parameters.
        Validates and normalizes incoming params (supports flat namespaces for Pydantic args).
        """
        # Convert string to Enum if needed
        if isinstance(generator_type, str):
            try:
                generator_type = DatasetGeneratorRegistry(generator_type)
            except ValueError:
                raise ValueError(f"Unknown generator type: {generator_type}")

        generator_class = self._registry.get(generator_type)
        if not generator_class:
            raise ValueError(f"Generator type {generator_type} not registered.")

        # Process config to handle grouped parameters (namespaces)
        sig = inspect.signature(generator_class.__init__)
        final_kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            # 1. Check if the parameter is a Pydantic model
            if is_pydantic_model(param.annotation):
                # If the parameter is a Pydantic model, try to instantiate it from the flat config
                if name in params and isinstance(params[name], dict):
                    # Explicitly provided as a sub-dictionary
                    final_kwargs[name] = param.annotation(**params[name])
                else:
                    # Otherwise, extract fields from the top-level config
                    model_fields = param.annotation.model_fields.keys()
                    filtered_config = {
                        k: v for k, v in params.items() if k in model_fields
                    }
                    if filtered_config or param.default is inspect.Parameter.empty:
                        final_kwargs[name] = param.annotation(**filtered_config)
            else:
                # 2. Simple parameter: normalize (Enums, basic types)
                if name in params:
                    final_kwargs[name] = normalize_value(params[name], param.annotation)
                elif param.default is not inspect.Parameter.empty:
                    final_kwargs[name] = param.default

        return generator_class(**final_kwargs)

    def get_generator_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of generator schemas for discovery purposes.
        """
        schemas = []
        for gen_id, gen_class in self._registry.items():
            parameters = []

            # Smart check: if the class is a Pydantic model, use its fields directly
            if is_pydantic_model(gen_class):
                for field_name, field in gen_class.model_fields.items():
                    parameters.extend(
                        inspect_parameter(
                            name=field_name,
                            annotation=field.annotation,
                            default=field.default
                            if field.default != ...
                            else inspect.Parameter.empty,
                            description=field.description,
                        )
                    )
            else:
                # Fallback to __init__ signature inspection (handles regular classes)
                sig = inspect.signature(gen_class.__init__)
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
                    "id": gen_id.value,
                    "name": gen_class.__name__.replace("Generator", " Generator"),
                    "parameters": parameters,
                }
            )

        return schemas
