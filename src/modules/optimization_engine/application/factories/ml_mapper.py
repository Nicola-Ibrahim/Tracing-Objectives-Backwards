from typing import Any

from ...domain.model_management.enums.ml_mapper_type import (
    MlMapperType,
)
from ...domain.model_management.interfaces.base_ml_mapper import (
    BaseMlMapper,
)
from ...infrastructure.ml.deterministic import (
    CloughTocherMlMapper,
    GaussianProcessMlMapper,
    KrigingMlMapper,
    LinearNDMlMapper,
    NearestNDMlMapper,
    NNMlMapper,
    RBFMlMapper,
    SplineMlMapper,
    SVRMlMapper,
)
from ...infrastructure.ml.probabilistic import (
    CVAEMlMapper,
    MDNMlMapper,
)


class MlMapperFactory:
    """
    Factory that uses a registry to dynamically create various interpolator instances.
    This design adheres to the Open/Closed Principle.
    """

    # A class-level registry to map the enum type to the corresponding class
    _registry: dict[MlMapperType, BaseMlMapper] = {
        MlMapperType.RBF_ND.value: RBFMlMapper,
        MlMapperType.LINEAR_ND.value: LinearNDMlMapper,
        MlMapperType.NEAREST_NEIGHBORS_ND.value: NearestNDMlMapper,
        MlMapperType.NEURAL_NETWORK_ND.value: NNMlMapper,
        MlMapperType.CLOUGH_TOCHER_ND.value: CloughTocherMlMapper,
        MlMapperType.GAUSSIAN_PROCESS_ND.value: GaussianProcessMlMapper,
        MlMapperType.SPLINE_ND.value: SplineMlMapper,
        MlMapperType.KRIGING_ND.value: KrigingMlMapper,
        MlMapperType.SVR_ND.value: SVRMlMapper,
        MlMapperType.CVAE_ND.value: CVAEMlMapper,
        MlMapperType.MDN_ND.value: MDNMlMapper,
    }

    def create(self, params: dict[str, Any]) -> BaseMlMapper:
        """
        Creates and returns an instance of an interpolator based on the specified type
        by looking it up in the factory's registry.

        Args:
            type (MlMapperType): The enum type indicating which interpolator
                                                  to create.
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseMlMapper: An initialized instance of the requested interpolator.

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
