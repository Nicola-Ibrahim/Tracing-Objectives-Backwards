from typing import Any

from ...domain.interpolation.entities.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ...domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from .splines.bivariate.cubic import CubicSplineInverseDecisionMapper
from .splines.bivariate.pchip import PchipInverseDecisionMapper
from .splines.bivariate.quadratic import QuadraticInverseDecisionMapper
from .splines.bivariate.rbf import RBFInverseDecisionMapper
from .splines.multivariate.linear import LinearNDInverseDecisionMapper
from .splines.multivariate.nearest_neighbors import NearestNDInverseDecisionMapper


class InverseDecisionMapperFactory:
    """
    This factory Manages the creation of various interpolator instances dynamically
    based on the InverseDecisionMapperType provided.
    """

    def create(
        self, type: InverseDecisionMapperType, params: dict[str, Any]
    ) -> BaseInverseDecisionMapper:
        """
        Creates and returns an instance of an interpolator based on the specified type.

        Args:
            type (InverseDecisionMapperType): The enum type indicating which interpolator
                                                  to create (e.g., InverseDecisionMapperType.NEURAL_NETWORK).
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseInverseDecisionMapper: An initialized instance of the requested interpolator,
                              conforming to the BaseInverseDecisionMapper interface.

        Raises:
            ValueError: If an unknown or unsupported InverseDecisionMapperType is provided.
        """

        match type:
            case InverseDecisionMapperType.QUADRATIC:
                return QuadraticInverseDecisionMapper(**params)
            case InverseDecisionMapperType.CUBIC:
                return CubicSplineInverseDecisionMapper(**params)
            case InverseDecisionMapperType.PCHIP:
                return PchipInverseDecisionMapper(**params)
            case InverseDecisionMapperType.RBF:
                return RBFInverseDecisionMapper(**params)
            case InverseDecisionMapperType.LINEAR_ND:
                return LinearNDInverseDecisionMapper(**params)
            case InverseDecisionMapperType.NEAREST_NEIGHBORS:
                return NearestNDInverseDecisionMapper(**params)
            case _:
                raise ValueError(f"Unknown interpolator type: {type}")
