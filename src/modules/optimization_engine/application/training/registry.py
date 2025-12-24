from ...domain.modeling.enums.estimator_key import EstimatorKeyEnum
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

ESTIMATOR_PARAM_REGISTRY: dict[EstimatorKeyEnum, type[EstimatorParamsBase]] = {
    EstimatorKeyEnum.COCO: COCOEstimatorParams,
    EstimatorKeyEnum.CVAE: CVAEEstimatorParams,
    EstimatorKeyEnum.GAUSSIAN_PROCESS: GaussianProcessEstimatorParams,
    EstimatorKeyEnum.MDN: MDNEstimatorParams,
    EstimatorKeyEnum.NEURAL_NETWORK: NeuralNetworkEstimatorParams,
    EstimatorKeyEnum.RBF: RBFEstimatorParams,
}


def default_metric_configs() -> list[ValidationMetricConfig]:
    return [
        ValidationMetricConfig(type=metric.value, params={})
        for metric in DEFAULT_VALIDATION_METRICS
    ]
