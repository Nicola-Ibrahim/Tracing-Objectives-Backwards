from typing import Any

from ...domain.model_management.enums.estimator_type_enum import (
    EstimatorTypeEnum,
)
from ...domain.model_management.interfaces.base_estimator import (
    BaseEstimator,
)
from ...infrastructure.ml.deterministic import (
    CloughTocherEstimator,
    GaussianProcessEstimator,
    KrigingEstimator,
    LinearNDEstimator,
    NearestNDEstimator,
    NNEstimator,
    RBFEstimator,
    SplineEstimator,
    SVREstimator,
)
from ...infrastructure.ml.probabilistic import (
    CVAEEstimator,
    MDNEstimator,
)


class EstimatorFactory:
    """
    Factory that uses a registry to dynamically create various interpolator instances.
    This design adheres to the Open/Closed Principle.
    """

    # A class-level registry to map the enum type to the corresponding class
    _registry: dict[EstimatorTypeEnum, BaseEstimator] = {
        EstimatorTypeEnum.RBF_ND.value: RBFEstimator,
        EstimatorTypeEnum.LINEAR_ND.value: LinearNDEstimator,
        EstimatorTypeEnum.NEAREST_NEIGHBORS_ND.value: NearestNDEstimator,
        EstimatorTypeEnum.NEURAL_NETWORK_ND.value: NNEstimator,
        EstimatorTypeEnum.CLOUGH_TOCHER_ND.value: CloughTocherEstimator,
        EstimatorTypeEnum.GAUSSIAN_PROCESS_ND.value: GaussianProcessEstimator,
        EstimatorTypeEnum.SPLINE_ND.value: SplineEstimator,
        EstimatorTypeEnum.KRIGING_ND.value: KrigingEstimator,
        EstimatorTypeEnum.SVR_ND.value: SVREstimator,
        EstimatorTypeEnum.CVAE_ND.value: CVAEEstimator,
        EstimatorTypeEnum.MDN_ND.value: MDNEstimator,
    }

    def create(self, params: dict[str, Any]) -> BaseEstimator:
        """
        Creates and returns an instance of an interpolator based on the specified type
        by looking it up in the factory's registry.

        Args:
            type (EstimatorTypeEnum): The enum type indicating which interpolator
                                                  to create.
            params (dict[str, Any]): A dictionary of parameters to pass to the
                                     interpolator's constructor.

        Returns:
            BaseEstimator: An initialized instance of the requested interpolator.

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
