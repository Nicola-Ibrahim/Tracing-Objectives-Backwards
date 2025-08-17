from ...domain.model_management.interfaces.base_metric import BaseValidationMetric
from ...infrastructure.metrics import MeanSquaredErrorValidationMetric


class MetricFactory:
    """
    Concrete factory for creating various validation metric instances.
    """

    def create(self, config: dict) -> BaseValidationMetric:
        """
        Creates and returns a concrete validation metric instance based on the given type and parameters.
        """
        metric_type = config.get("type")
        params = config.get("params", {})

        if metric_type == "MSE":
            return MeanSquaredErrorValidationMetric(**params)

        # elif metric_type == "MAE":
        #     return MeanAbsoluteErrorValidationMetric(**params)

        # elif metric_type == "R2":
        #     return R2ScoreValidationMetric(**params)

        # elif metric_type == "MAPE":
        #     return MeanAbsolutePercentageErrorValidationMetric(**params)

        # elif metric_type == "SMAPE":
        #     return SymmetricMeanAbsolutePercentageErrorValidationMetric(**params)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
