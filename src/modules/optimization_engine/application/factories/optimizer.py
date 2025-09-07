from ...domain.datasets.interfaces.base_optimizer import BaseOptimizer
from ...infrastructure.generation.optimizers.minimizer import Minimizer, MinimizerConfig


class OptimizerFactory:
    _registry = {
        "minimizer": lambda problem, algorithm, config: Minimizer(
            problem=problem,
            algorithm=algorithm,
            config=config,
        ),
    }

    def create(self, config: dict, **kwargs) -> BaseOptimizer:
        """Creates an optimizer instance using the factory registry.

        The config must include a 'type' key. If not provided, 'minimizer' is used.
        """
        optimizer_type = config.get("type", "minimizer")

        try:
            optimizer_ctor = self._registry[optimizer_type]
        except KeyError as e:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}") from e

        return optimizer_ctor(config=MinimizerConfig(**config), **kwargs)
