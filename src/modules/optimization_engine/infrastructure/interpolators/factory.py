from typing import Any

from ...domain.interpolation.entities.interpolator_type import (
    InterpolatorType,
)
from ...domain.interpolation.interfaces.base_interpolator import BaseInterpolator
from .geodesic import GeodesicInterpolator
from .knn import KNearestNeighborInterpolator
from .linear import LinearInterpolator
from .nn import NeuralNetworkInterpolator


class InterpolatorFactory:
    """
    This factory Manages the creation of various interpolator instances dynamically
    based on the InterpolatorType provided.
    """

    def create_interpolator(
        self, interpolator_type: InterpolatorType, params: dict[str, Any]
    ) -> BaseInterpolator:
        """
        Creates and returns an instance of an interpolator based on the specified type.

        Args:
            interpolator_type (InterpolatorType): The enum type indicating which interpolator
                                                  to create (e.g., InterpolatorType.NEURAL_NETWORK).
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseInterpolator: An initialized instance of the requested interpolator,
                              conforming to the BaseInterpolator interface.

        Raises:
            ValueError: If an unknown or unsupported InterpolatorType is provided.
        """
        if interpolator_type == InterpolatorType.NEURAL_NETWORK:
            return NeuralNetworkInterpolator(**params)

        elif interpolator_type == InterpolatorType.GEODESIC:
            return GeodesicInterpolator(**params)

        elif interpolator_type == InterpolatorType.K_NEAREST_NEIGHBOR:
            return KNearestNeighborInterpolator(**params)

        elif interpolator_type == InterpolatorType.LINEAR:
            return LinearInterpolator(**params)

        else:
            raise ValueError(f"Unknown interpolator type: {interpolator_type}")
