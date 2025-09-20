from ...domain.modeling.enums.normalizer_type import NormalizerTypeEnum
from ...domain.modeling.interfaces.base_normalizer import BaseNormalizer
from ...infrastructure.normalizers import (
    HypercubeNormalizer,
    LogNormalizer,
    MinMaxScalerNormalizer,
    StandardNormalizer,
    UnitVectorNormalizer,
)


class NormalizerFactory:
    """
    Concrete factory for creating various normalizer instances.
    """

    _registry = {
        NormalizerTypeEnum.MIN_MAX: MinMaxScalerNormalizer,
        NormalizerTypeEnum.HYPERCUBE: HypercubeNormalizer,
        NormalizerTypeEnum.STANDARD: StandardNormalizer,
        NormalizerTypeEnum.UNIT_VECTOR: UnitVectorNormalizer,
        NormalizerTypeEnum.LOG: LogNormalizer,
    }

    def create(self, config: dict) -> BaseNormalizer:
        """
        Creates and returns a concrete normalizer instance based on the given type and parameters.
        """
        normalizer_type = config.get("type")
        params = config.get("params", {})

        try:
            normalizer_ctor = self._registry[normalizer_type]
        except KeyError as e:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}") from e

        return normalizer_ctor(**params)
