from ...domain.model_management.interfaces.base_normalizer import BaseNormalizer
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

    def create(self, config: dict) -> BaseNormalizer:
        """
        Creates and returns a concrete normalizer instance based on the given type and parameters.
        """
        normalizer_type = config.get("type")
        params = config.get("params", {})

        if normalizer_type == "MinMaxScaler":
            return MinMaxScalerNormalizer(**params)
        elif normalizer_type == "HypercubeNormalizer":
            return HypercubeNormalizer(**params)
        elif normalizer_type == "StandardNormalizer":
            return StandardNormalizer(**params)
        elif normalizer_type == "UnitVectorNormalizer":
            return UnitVectorNormalizer(**params)
        elif normalizer_type == "LogNormalizer":
            return LogNormalizer(**params)

        else:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")
