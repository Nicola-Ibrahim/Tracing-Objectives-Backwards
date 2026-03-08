import inspect
from typing import Any, Dict, List, Type

from ....shared.infrastructure.inspection import inspect_parameter
from ...domain.enums.transform_type import TransformTypeEnum
from ...domain.interfaces.base_transform import BaseTransformer
from ...infrastructure.normalizers import (
    HypercubeNormalizer,
    LogNormalizer,
    MinMaxScalerNormalizer,
    StandardNormalizer,
    UnitVectorNormalizer,
)


class TransformerFactory:
    """
    Concrete factory for creating BaseTransformer instances.
    """

    _registry: Dict[TransformTypeEnum, Type[BaseTransformer]] = {
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
                raise ValueError(f"Unknown transform type: {transform_type}")

        step_cls = cls._registry.get(transform_type)
        if not step_cls:
            raise NotImplementedError(
                f"Instantiation for transform type '{transform_type}' is not yet supported."
            )

        params = config.get("params", {})
        return step_cls(**params)

    @classmethod
    def from_checkpoint(cls, config: dict, state: dict) -> BaseTransformer:
        transform_type = config.get("type")
        if isinstance(transform_type, str):
            try:
                transform_type = TransformTypeEnum(transform_type)
            except ValueError:
                raise ValueError(f"Unknown transform type: {transform_type}")

        step_cls = cls._registry.get(transform_type)
        if not step_cls:
            raise NotImplementedError(
                f"Instantiation for transform type '{transform_type}' is not yet supported."
            )

        return step_cls.from_checkpoint(config, state)

    def get_transformer_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns metadata for all registered transformers using inspection.
        """
        schemas = []
        for transform_type, transformer_class in self._registry.items():
            sig = inspect.signature(transformer_class.__init__)
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
                    "type": transform_type.value,
                    "name": transform_type.value.replace("_", " ").title(),
                    "parameters": parameters,
                }
            )

        return schemas
