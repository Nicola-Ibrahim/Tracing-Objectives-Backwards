from typing import Any, Type

from ...domain.interpolation.entities.inverse_decision_mapper_type import (
    InverseDecisionMapperType,
)
from ...domain.interpolation.interfaces.base_inverse_decision_mappers import (
    BaseInverseDecisionMapper,
)
from .multivariate.clough_tocher import CloughTocherInverseDecisionMapper
from .multivariate.linear import LinearNDInverseDecisionMapper
from .multivariate.nearest_neighbors import NearestNDInverseDecisionMapper
from .multivariate.nn import NeuralNetworkInterpolator
from .multivariate.rbf import RBFInverseDecisionMapper


class InverseDecisionMapperFactory:
    """
    Factory that uses a registry to dynamically create various interpolator instances.
    This design adheres to the Open/Closed Principle.
    """

    # A class-level registry to map the enum type to the corresponding class
    _registry: dict[InverseDecisionMapperType, Type[BaseInverseDecisionMapper]] = {
        InverseDecisionMapperType.RBF_ND: RBFInverseDecisionMapper,
        InverseDecisionMapperType.LINEAR_ND: LinearNDInverseDecisionMapper,
        InverseDecisionMapperType.NEAREST_NEIGHBORS_ND: NearestNDInverseDecisionMapper,
        InverseDecisionMapperType.NEURAL_NETWORK_ND: NeuralNetworkInterpolator,
        InverseDecisionMapperType.CLOUGH_TOCHER_ND: CloughTocherInverseDecisionMapper,
    }

    def create(
        self, type: InverseDecisionMapperType, params: dict[str, Any]
    ) -> BaseInverseDecisionMapper:
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
        mapper_class = self._registry.get(type)

        if mapper_class is None:
            raise ValueError(f"Unknown or unsupported interpolator type: {type.value}")

        # Instantiate the class using the parameters from the dictionary
        return mapper_class(**params)

    @classmethod
    def register(
        cls,
        type: InverseDecisionMapperType,
        mapper_class: Type[BaseInverseDecisionMapper],
    ) -> None:
        """
        Registers a new interpolator class with the factory's registry.
        This allows for extending the factory's capabilities without modifying the create() method.
        """
        if not issubclass(mapper_class, BaseInverseDecisionMapper):
            raise TypeError(
                "mapper_class must be a subclass of BaseInverseDecisionMapper"
            )
        if type in cls._registry:
            print(f"Warning: Overwriting existing mapper for type '{type.value}'.")
        cls._registry[type] = mapper_class
