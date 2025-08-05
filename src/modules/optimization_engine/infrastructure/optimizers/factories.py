from ...domain.generation.interfaces.base_algorithm import BaseAlgorithm
from ...domain.generation.interfaces.base_optimizer import BaseOptimizer
from ...domain.generation.interfaces.base_problem import BaseProblem
from ...infrastructure.optimizers.minimizer import Minimizer, MinimizerConfig

# Registry for optimizers
OPTIMIZER_REGISTRY = {
    "minimizer": Minimizer,
}


class OptimizerFactory:
    def create(
        self,
        config: dict,
        problem: BaseProblem,
        algorithm: BaseAlgorithm,
    ) -> BaseOptimizer:
        """
        Creates an optimizer instance from a type string and configuration data.
        """
        try:
            optimizer_class = OPTIMIZER_REGISTRY[config.get("type", "minimizer")]

            return optimizer_class(
                problem=problem,
                algorithm=algorithm,
                config=MinimizerConfig(**config),
            )

        except KeyError:
            raise ValueError(f"Available types are: {list(OPTIMIZER_REGISTRY.keys())}")

        except Exception as e:
            raise ValueError(
                f"Failed to create optimizer of type '{optimizer_class}' "
                f"with config '{config}'. Error: {e}"
            )
