from ...domain.generation.interfaces.base_optimizer import BaseOptimizer
from ...infrastructure.optimizers.minimizer import Minimizer, MinimizerConfig


class OptimizerFactory:
    _registery = {
        "minimizer": lambda problem, algorithm, config: Minimizer(
            problem=problem,
            algorithm=algorithm,
            config=config,
        ),
    }

    def create(self, config: dict, **kwargs) -> BaseOptimizer:
        """
        Creates an optimizer instance from a type string and configuration data.
        """
        optimizer_type = config.get("type", "minimizer")

        if optimizer_type not in self._registery:
            raise ValueError("Optimizer type must be specified in the config.")

        optimizer_class = self._registery[optimizer_type]

        return optimizer_class(config=MinimizerConfig(**config), **kwargs)
