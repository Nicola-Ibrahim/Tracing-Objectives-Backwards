from typing import Any, Type

from ....shared.infrastructure.discovery import (
    build_constructor_kwargs,
    extract_constructor_schema,
)
from ...domain.enums.generator_type import DatasetGeneratorRegistry
from ...domain.interfaces.base_data_source import BaseDataSource
from .pymoo.generate import PymooGenerator


class DataGeneratorFactory:
    _registry: dict[DatasetGeneratorRegistry, Type[BaseDataSource]] = {
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

        # Build constructor arguments using centralized logic
        final_kwargs = build_constructor_kwargs(generator_class, params)

        return generator_class(**final_kwargs)

    def get_generator_schemas(self) -> list[dict[str, Any]]:
        """
        Returns a list of generator schemas for discovery purposes.
        """
        schemas = []
        for gen_id, gen_class in self._registry.items():
            schemas.append(
                {
                    "type": gen_id.value,
                    "name": gen_class.__name__.replace("Generator", " Generator"),
                    "parameters": extract_constructor_schema(gen_class),
                }
            )

        return schemas
