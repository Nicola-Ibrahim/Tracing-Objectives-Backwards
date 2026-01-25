from ...modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ...modeling.domain.enums.metric_key import DefaultValidationMetricEnum
from ...modeling.domain.value_objects.estimator_params import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParamsBase,
    GaussianProcessEstimatorParams,
    INNEstimatorParams,
    MDNEstimatorParams,
    NearestNeighborsEstimatorParams,
    NeuralNetworkEstimatorParams,
    RBFEstimatorParams,
    ValidationMetricConfig,
)

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


def default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=metric.value, params={})
        for metric in DEFAULT_VALIDATION_METRICS
    ]
