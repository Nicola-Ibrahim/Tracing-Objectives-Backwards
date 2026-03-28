from typing import Any, Type

from ....shared.infrastructure.discovery import (
    build_constructor_kwargs,
    extract_constructor_schema,
)
from ...domain.enums.transform_type import TransformTypeEnum
from ...domain.interfaces.base_transform import BaseTransformer
from ...domain.services.transformation_domain_service import ITransformerFactory
from ...infrastructure.normalizers import (
    HypercubeNormalizer,
    LogNormalizer,
    MinMaxScalerNormalizer,
    StandardNormalizer,
    UnitVectorNormalizer,
)


class TransformerFactory(ITransformerFactory):
    """
    Concrete factory for creating BaseTransformer instances.
    """

    _registry: dict[TransformTypeEnum, Type[BaseTransformer]] = {
        TransformTypeEnum.MIN_MAX: MinMaxScalerNormalizer,
        TransformTypeEnum.STANDARD: StandardNormalizer,
        TransformTypeEnum.LOG: LogNormalizer,
        TransformTypeEnum.HYPERCUBE: HypercubeNormalizer,
        TransformTypeEnum.UNIT_VECTOR: UnitVectorNormalizer,
    }

    @classmethod
    def create(cls, config: dict) -> BaseTransformer:
        """
        Creates and returns a concrete BaseTransformer instance.
        """
        transform_type = config.get("type")
        if isinstance(transform_type, str):
            try:
                transform_type = TransformTypeEnum(transform_type)
            except ValueError:
                raise ValueError(f"Unknown transform type: {transform_type}") from None

        step_cls = cls._registry.get(transform_type)
        if not step_cls:
            raise NotImplementedError(
                f"Transform type '{transform_type}' is not yet supported."
            )

        params = config.get("params", {})
        # Build constructor arguments using centralized logic
        final_kwargs = build_constructor_kwargs(step_cls, params)

        return step_cls(**final_kwargs)

    @classmethod
    def from_checkpoint(cls, config: dict, state: dict) -> BaseTransformer:
        transform_type = config.get("type")
        if isinstance(transform_type, str):
            try:
                transform_type = TransformTypeEnum(transform_type)
            except ValueError:
                raise ValueError(f"Unknown transform type: {transform_type}") from None

        step_cls = cls._registry.get(transform_type)
        if not step_cls:
            raise NotImplementedError(
                f"Transform type '{transform_type}' is not yet supported."
            )

        return step_cls.from_checkpoint(config, state)

    def get_transformer_schemas(self) -> list[dict[str, Any]]:
        """
        Returns metadata for all registered transformers using inspection.
        """
        schemas = []
        for transform_type, transformer_class in self._registry.items():
            schemas.append(
                {
                    "type": transform_type.value,
                    "name": transform_type.value.replace("_", " ").title(),
                    "parameters": extract_constructor_schema(transformer_class),
                }
            )

        return schemas
