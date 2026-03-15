from typing import Annotated, Any, Callable, Dict, Union

from pydantic import Field

from ...domain.enums.estimator_type import (
    EstimatorTypeEnum,
)
from ...domain.enums.metric_key import DefaultValidationMetricEnum
from ...domain.interfaces.base_estimator import (
    BaseEstimator,
)
from ...domain.value_objects.estimator_params import EstimatorParamsBase
from ...infrastructure.estimators.deterministic import (
    GaussianProcessEstimator,
    NearestNDEstimator,
    NNEstimator,
    RBFEstimator,
)
from ...infrastructure.estimators.deterministic.coco_biobj_function import (
    COCOEstimator,
    COCOEstimatorParams,
)
from ...infrastructure.estimators.deterministic.gaussian_process import (
    GaussianProcessEstimatorParams,
)
from ...infrastructure.estimators.deterministic.nearest_neighbors import (
    NearestNeighborsEstimatorParams,
)
from ...infrastructure.estimators.deterministic.nn import (
    NeuralNetworkEstimatorParams,
)
from ...infrastructure.estimators.deterministic.rbf import RBFEstimatorParams
from ...infrastructure.estimators.probabilistic import (
    CVAEEstimator,
    INNEstimator,
    MDNEstimator,
)
from ...infrastructure.estimators.probabilistic.cvae import CVAEEstimatorParams
from ...infrastructure.estimators.probabilistic.inn import INNEstimatorParams
from ...infrastructure.estimators.probabilistic.mdn import MDNEstimatorParams

DEFAULT_VALIDATION_METRICS: tuple[DefaultValidationMetricEnum, ...] = (
    DefaultValidationMetricEnum.MSE,
    DefaultValidationMetricEnum.MAE,
    DefaultValidationMetricEnum.R2,
)

ESTIMATOR_PARAM_REGISTRY: dict[EstimatorTypeEnum, type[EstimatorParamsBase]] = {
    EstimatorTypeEnum.COCO: COCOEstimatorParams,
    EstimatorTypeEnum.CVAE: CVAEEstimatorParams,
    EstimatorTypeEnum.GAUSSIAN_PROCESS_ND: GaussianProcessEstimatorParams,
    EstimatorTypeEnum.INN: INNEstimatorParams,
    EstimatorTypeEnum.MDN: MDNEstimatorParams,
    EstimatorTypeEnum.NEAREST_NEIGHBORS_ND: NearestNeighborsEstimatorParams,
    EstimatorTypeEnum.NEURAL_NETWORK_ND: NeuralNetworkEstimatorParams,
    EstimatorTypeEnum.RBF: RBFEstimatorParams,
}

EstimatorParams = Annotated[
    Union[
        COCOEstimatorParams,
        NeuralNetworkEstimatorParams,
        NearestNeighborsEstimatorParams,
        RBFEstimatorParams,
        GaussianProcessEstimatorParams,
        MDNEstimatorParams,
        CVAEEstimatorParams,
        INNEstimatorParams,
    ],
    Field(discriminator="type"),
]


class EstimatorFactory:
    """
    Factory that uses a registry to dynamically create various interpolator instances.
    This design adheres to the Open/Closed Principle.
    """

    # A class-level registry to map the enum type to the corresponding class
    _registry: dict[EstimatorTypeEnum, type[BaseEstimator]] = {
        EstimatorTypeEnum.RBF: RBFEstimator,
        EstimatorTypeEnum.NEAREST_NEIGHBORS_ND: NearestNDEstimator,
        EstimatorTypeEnum.NEURAL_NETWORK_ND: NNEstimator,
        EstimatorTypeEnum.GAUSSIAN_PROCESS_ND: GaussianProcessEstimator,
        EstimatorTypeEnum.CVAE: CVAEEstimator,
        EstimatorTypeEnum.MDN: MDNEstimator,
        EstimatorTypeEnum.COCO: COCOEstimator,
        EstimatorTypeEnum.INN: INNEstimator,
    }

    # -------- Forward models --------
    _forward_registry: Dict[str, Callable[..., BaseEstimator]] = {
        "coco_biobj": lambda **p: COCOEstimator(
            params=ESTIMATOR_PARAM_REGISTRY[EstimatorTypeEnum.COCO](**p)
        ),
    }

    def create(
        self, type: EstimatorTypeEnum | str, params: EstimatorParamsBase | dict
    ) -> BaseEstimator:
        """
        Creates and returns an instance of an interpolator based on the specified type.
        Supports both direct Enum usage or string (case-insensitive value lookup).
        Handles both Pydantic models and raw dicts as params.
        """
        # 1. Normalize type to Enum
        if isinstance(type, str):
            try:
                # Try case-insensitive matching against values (e.g., "mdn" or "MDN" -> MDN)
                type_enum = EstimatorTypeEnum(type.lower())
            except ValueError:
                # Fallback: check if the string matches the member name (case-insensitive)
                try:
                    type_enum = EstimatorTypeEnum[type.upper()]
                except KeyError:
                    raise ValueError(f"Unknown estimator type: {type}")
        else:
            type_enum = type

        # 2. Get the correct parameter model
        params_model = ESTIMATOR_PARAM_REGISTRY.get(type_enum)
        if params_model is None:
            raise ValueError(f"Unsupported estimator params type: {type_enum!r}")

        # 3. Handle parameter validation/coercion
        if isinstance(params, EstimatorParamsBase):
            # If already an instance of the target model, use it; otherwise coerce
            if isinstance(params, params_model):
                estimator_params = params
            else:
                estimator_params = params_model.model_validate(params.model_dump())
        else:
            # Dictionary input: filter to allowed fields and validate
            allowed = set(params_model.model_fields.keys())
            filtered = {k: v for k, v in params.items() if k in allowed}
            estimator_params = params_model.model_validate(filtered)

        # 4. Lookup constructor and instantiate
        try:
            mapper_ctor = self._registry[type_enum]
        except KeyError as e:
            raise ValueError(
                f"Estimator type {type_enum} is not implemented in factory registry"
            ) from e

        return mapper_ctor(params=estimator_params)

    @classmethod
    def from_checkpoint(cls, config: dict[str, Any]) -> BaseEstimator:
        """Create an estimator from a checkpoint."""
        estimator_type = config.get("type")
        if not estimator_type:
            raise ValueError("Estimator config must include 'type'.")
        try:
            ctor = cls._registry[estimator_type]
        except KeyError as e:
            raise ValueError(f"Unknown estimator type: {estimator_type}") from e
        params = {k: v for k, v in config.items() if k != "type"}
        return ctor.from_checkpoint(**params)

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
