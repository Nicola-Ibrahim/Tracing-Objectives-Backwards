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
        """Creates a validation metric instance using the factory registry.

        The config must contain a 'type' key. If the type is not registered
        a ValueError is raised.
        """
        metric_type = config.get("type")
        params = config.get("params", {})

        try:
            metric_ctor = self._registry[metric_type]
        except KeyError as e:
            raise ValueError(f"Unknown metric type: {metric_type}") from e

        return metric_ctor(**params)

    def create_multiple(self, configs: list[dict]) -> list[BaseValidationMetric]:
        """Creates multiple validation metric instances from a list of configs."""

        metrics = []
        for config in configs:
            metrics.append(self.create(config))
        return metrics
