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

    _registry = {
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
        If `state` is provided, it reconstructs a fitted instance.
        Otherwise, it initializes a fresh instance from configuration parameters.
        """
        transform_type = config.get("type")

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

        step_cls = cls._registry.get(transform_type)
        if not step_cls:
            raise NotImplementedError(
                f"Instantiation for transform type '{transform_type}' is not yet supported."
            )

        return step_cls.from_checkpoint(config, state)
