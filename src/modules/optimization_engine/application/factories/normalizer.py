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

    _registry = {
        "MinMaxScaler": MinMaxScalerNormalizer,
        "HypercubeNormalizer": HypercubeNormalizer,
        "StandardNormalizer": StandardNormalizer,
        "UnitVectorNormalizer": UnitVectorNormalizer,
        "LogNormalizer": LogNormalizer,
    }

    def create(self, config: dict) -> BaseNormalizer:
        """
        Creates and returns a concrete normalizer instance based on the given type and parameters.
        """
        normalizer_type = config.get("type")

        if not normalizer_type:
            raise ValueError("Normalizer type must be specified in the config.")

        params = config.get("params", {})

        if normalizer_type not in self._registry:
            raise ValueError(f"Unknown normalizer type: {normalizer_type}")

        return self._registry[normalizer_type](**params)
