from ...domain.enums.normalizer_type import NormalizerTypeEnum
from ...domain.interfaces.base_transform import TransformTarget
from ...infrastructure.normalizers import NormalizationStep


class NormalizerFactory:
    """
    Concrete factory for creating NormalizationStep instances.
    """

    def create(self, config: dict, target: TransformTarget) -> NormalizationStep:
        """
        Creates and returns a concrete NormalizationStep instance based on type and parameters.
        """
        normalizer_type = config.get("type", "min_max")
        params = config.get("params", {})

        # We assume NormalizerTypeEnum can convert from string
        try:
            norm_enum = NormalizerTypeEnum(normalizer_type)
        except ValueError as e:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}") from e

        return NormalizationStep(
            target=target, normalizer_type=norm_enum, params=params
        )
