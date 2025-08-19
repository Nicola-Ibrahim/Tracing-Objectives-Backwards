from ...domain.model_evaluation.interfaces.base_metric import BaseValidationMetric
from ...infrastructure.metrics import (
    MeanAbsoluteErrorValidationMetric,
    MeanSquaredErrorValidationMetric,
    R2ScoreValidationMetric,
)


class MetricFactory:
    """
    Concrete factory for creating various validation metric instances.
    """

    _registry = {
        "MSE": MeanSquaredErrorValidationMetric,
        "MAE": MeanAbsoluteErrorValidationMetric,
        "R2": R2ScoreValidationMetric,
    }

    def create(self, config: dict) -> BaseValidationMetric:
        """
        Creates and returns a concrete validation metric instance based on the given type and parameters.
        """
        metric_type = config.get("type")
        params = config.get("params", {})

        if metric_type in self._registry:
            return self._registry[metric_type](**params)

        raise ValueError(f"Unknown metric type: {metric_type}")
