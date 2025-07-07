from typing import Any

from ...domain.interpolation.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ...domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from .deterministic.clough_tocher import CloughTocherInverseDecisionMapper
from .deterministic.gaussian_process import GaussianProcessInverseDecisionMapper
from .deterministic.kriging import KrigingInverseDecisionMapper
from .deterministic.linear import LinearNDInverseDecisionMapper
from .deterministic.nearest_neighbors import NearestNDInverseDecisionMapper
from .deterministic.nn import NeuralNetworkInterpolator
from .deterministic.rbf import RBFInverseDecisionMapper
from .deterministic.spline import SplineInverseDecisionMapper
from .deterministic.svr import SVRInverseDecisionMapper
from .probabilistic.cvae import CVAEInverseDecisionMapper
from .probabilistic.mdn import MDNInverseDecisionMapper


class InverseDecisionMapperFactory:
    """
    Factory that uses a registry to dynamically create various interpolator instances.
    This design adheres to the Open/Closed Principle.
    """

    # A class-level registry to map the enum type to the corresponding class
    _registry: dict[InverseDecisionMapperType, BaseInverseDecisionMapper] = {
        InverseDecisionMapperType.RBF_ND.value: RBFInverseDecisionMapper,
        InverseDecisionMapperType.LINEAR_ND.value: LinearNDInverseDecisionMapper,
        InverseDecisionMapperType.NEAREST_NEIGHBORS_ND.value: NearestNDInverseDecisionMapper,
        InverseDecisionMapperType.NEURAL_NETWORK_ND.value: NeuralNetworkInterpolator,
        InverseDecisionMapperType.CLOUGH_TOCHER_ND.value: CloughTocherInverseDecisionMapper,
        InverseDecisionMapperType.GAUSSIAN_PROCESS_ND.value: GaussianProcessInverseDecisionMapper,
        InverseDecisionMapperType.SPLINE_ND.value: SplineInverseDecisionMapper,
        InverseDecisionMapperType.KRIGING_ND.value: KrigingInverseDecisionMapper,
        InverseDecisionMapperType.SVR_ND.value: SVRInverseDecisionMapper,
        InverseDecisionMapperType.CVAE_ND.value: CVAEInverseDecisionMapper,
        InverseDecisionMapperType.MDN_ND.value: MDNInverseDecisionMapper,
    }

    def create(self, params: dict[str, Any]) -> BaseInverseDecisionMapper:
        """
        Creates and returns an instance of an interpolator based on the specified type
        by looking it up in the factory's registry.

        Args:
            type (InverseDecisionMapperType): The enum type indicating which interpolator
                                                  to create.
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseInverseDecisionMapper: An initialized instance of the requested interpolator.

        Raises:
            ValueError: If the provided type is not registered in the factory.
        """
        # Look up the class in the registry dictionary
        if "type" not in params:
            raise ValueError(
                "The 'type' key must be present in the parameters dictionary."
            )

        mapper_class_type = params.pop("type", None)

        mapper_class = self._registry.get(mapper_class_type)

        if mapper_class is None:
            raise ValueError(
                f"Unknown or unsupported interpolator type: {mapper_class_type}"
            )

        # Instantiate the class using the parameters from the dictionary
        return mapper_class(**params)
