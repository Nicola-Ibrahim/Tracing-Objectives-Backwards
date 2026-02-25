from typing import Annotated, Union

from pydantic import Field

from ...modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modeling.domain.enums.metric_key import DefaultValidationMetricEnum
from ...modeling.domain.value_objects.estimator_params import (
    EstimatorParamsBase,
    ValidationMetricConfig,
)
from ...modeling.infrastructure.estimators.deterministic.coco_biobj_function import (
    COCOEstimatorParams,
)
from ...modeling.infrastructure.estimators.deterministic.gaussian_process import (
    GaussianProcessEstimatorParams,
)
from ...modeling.infrastructure.estimators.deterministic.nearest_neighbors import (
    NearestNeighborsEstimatorParams,
)
from ...modeling.infrastructure.estimators.deterministic.nn import (
    NeuralNetworkEstimatorParams,
)
from ...modeling.infrastructure.estimators.deterministic.rbf import RBFEstimatorParams
from ...modeling.infrastructure.estimators.probabilistic.cvae import CVAEEstimatorParams
from ...modeling.infrastructure.estimators.probabilistic.inn import INNEstimatorParams
from ...modeling.infrastructure.estimators.probabilistic.mdn import MDNEstimatorParams

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


def default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=metric.value, params={})
        for metric in DEFAULT_VALIDATION_METRICS
    ]
