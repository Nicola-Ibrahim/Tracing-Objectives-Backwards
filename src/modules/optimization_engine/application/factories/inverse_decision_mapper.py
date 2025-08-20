from typing import Any

from ...domain.model_management.enums.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ...domain.model_management.interfaces.base_inverse_decision_mapper import (
    BaseInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.deterministic import (
    CloughTocherInverseDecisionMapper,
    GaussianProcessInverseDecisionMapper,
    KrigingInverseDecisionMapper,
    LinearNDInverseDecisionMapper,
    NearestNDInverseDecisionMapper,
    NNInverseDecisionMapper,
    RBFInverseDecisionMapper,
    SplineInverseDecisionMapper,
    SVRInverseDecisionMapper,
)
from ...infrastructure.inverse_decision_mappers.probabilistic import (
    CVAEInverseDecisionMapper,
    MDNInverseDecisionMapper,
)


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
        InverseDecisionMapperType.NEURAL_NETWORK_ND.value: NNInverseDecisionMapper,
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

        mapper_class_type = params.get("type")

        try:
            mapper_ctor = self._registry[mapper_class_type]
        except KeyError as e:
            raise ValueError(
                f"Unknown or unsupported interpolator type: {mapper_class_type}"
            ) from e

        # Create a shallow copy of params without mutating the caller's dict
        ctor_params = {k: v for k, v in params.items() if k != "type"}
        return mapper_ctor(**ctor_params)
