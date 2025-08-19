from ...domain.generation.interfaces.base_algorithm import BaseAlgorithm
from ...infrastructure.algorithms.nsga2 import NSGAII, NSGA2Config


class AlgorithmFactory:
    _registry = {
        "nsga2": lambda config: NSGAII(NSGA2Config(**config)),
    }

    def create(self, config: dict) -> BaseAlgorithm:
        algo_type = config.pop("type", None)

        if algo_type.lower() not in self._registry:
            raise ValueError(f"Unsupported algorithm: {algo_type}")

        return self._registry[algo_type.lower()](config)
