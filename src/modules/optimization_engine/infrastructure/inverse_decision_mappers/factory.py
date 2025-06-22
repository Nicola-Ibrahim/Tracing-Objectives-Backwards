from typing import Any

from ...domain.interpolation.entities.interpolator_type import (
    InterpolatorType,
)
from ...domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from .ml.knn import KNearestNeighborInterpolator
from .nn import NeuralNetworkInterpolator
from .splines.geodesic import GeodesicInterpolator
from .splines.linear import LinearInterpolator


class InterpolatorFactory:
    """
    This factory Manages the creation of various interpolator instances dynamically
    based on the InterpolatorType provided.
    """

    def create_interpolator(
        self, interpolator_type: InterpolatorType, params: dict[str, Any]
    ) -> BaseInverseDecisionMapper:
        """
        Creates and returns an instance of an interpolator based on the specified type.

        Args:
            interpolator_type (InterpolatorType): The enum type indicating which interpolator
                                                  to create (e.g., InterpolatorType.NEURAL_NETWORK).
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseInverseDecisionMapper: An initialized instance of the requested interpolator,
                              conforming to the BaseInverseDecisionMapper interface.

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
