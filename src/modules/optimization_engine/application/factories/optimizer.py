from ...domain.generation.interfaces.base_optimizer import BaseOptimizer
from ...infrastructure.optimizers.minimizer import Minimizer, MinimizerConfig

# Registry for optimizers
OPTIMIZER_REGISTRY = {
    "minimizer": lambda problem, algorithm, config: Minimizer(
        problem=problem,
        algorithm=algorithm,
        config=config,
    ),
}


class OptimizerFactory:
    def create(self, config: dict, **kwargs) -> BaseOptimizer:
        """
        Creates an optimizer instance from a type string and configuration data.
        """
        optimizer_class = OPTIMIZER_REGISTRY[config.get("type", "minimizer")]

        return optimizer_class(config=MinimizerConfig(**config), **kwargs)
