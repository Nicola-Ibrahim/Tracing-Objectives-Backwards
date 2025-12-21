from typing import Any, Callable, Dict

from ...domain.modeling.enums.estimator_type import (
    EstimatorTypeEnum,
)
from ...domain.modeling.interfaces.base_estimator import (
    BaseEstimator,
)
from ...infrastructure.modeling.estimators.deterministic import (
    GaussianProcessEstimator,
    NearestNDEstimator,
    NNEstimator,
    RBFEstimator,
)
from ...infrastructure.modeling.estimators.deterministic.coco_biobj_function import (
    COCOEstimator,
)
from ...infrastructure.modeling.estimators.probabilistic import (
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
        EstimatorTypeEnum.RBF.value: RBFEstimator,
        EstimatorTypeEnum.NEAREST_NEIGHBORS_ND.value: NearestNDEstimator,
        EstimatorTypeEnum.NEURAL_NETWORK_ND.value: NNEstimator,
        EstimatorTypeEnum.GAUSSIAN_PROCESS_ND.value: GaussianProcessEstimator,
        EstimatorTypeEnum.CVAE.value: CVAEEstimator,
        EstimatorTypeEnum.MDN.value: MDNEstimator,
        EstimatorTypeEnum.COCO.value: COCOEstimator,
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

    # -------- Forward models --------
    _forward_registry: Dict[str, Callable[..., BaseEstimator]] = {
        "coco_biobj": lambda **p: COCOEstimator(**p),
    }

    def create_forward(self, config: dict[str, Any]) -> BaseEstimator:
        """Create a forward objective evaluator (forward decision mapper).

        Example config: {"type": "coco_biobj", "function_indices": 5}
        """
        ftype = config.get("type")
        if not ftype:
            raise ValueError("Forward model config must include 'type'.")
        try:
            ctor = self._forward_registry[ftype]
        except KeyError as e:
            raise ValueError(f"Unknown forward model type: {ftype}") from e
        params = {k: v for k, v in config.items() if k != "type"}
        return ctor(**params)
