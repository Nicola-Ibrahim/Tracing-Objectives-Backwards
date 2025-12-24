from ...domain.modeling.enums.estimator_type import EstimatorTypeEnum
from ...domain.modeling.enums.metric_key import DefaultValidationMetricEnum
from ...domain.modeling.value_objects.estimator_params import (
    COCOEstimatorParams,
    CVAEEstimatorParams,
    EstimatorParamsBase,
    GaussianProcessEstimatorParams,
    MDNEstimatorParams,
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
    EstimatorTypeEnum.MDN: MDNEstimatorParams,
    EstimatorTypeEnum.NEURAL_NETWORK_ND: NeuralNetworkEstimatorParams,
    EstimatorTypeEnum.RBF: RBFEstimatorParams,
}


def default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=metric.value, params={})
        for metric in DEFAULT_VALIDATION_METRICS
    ]
