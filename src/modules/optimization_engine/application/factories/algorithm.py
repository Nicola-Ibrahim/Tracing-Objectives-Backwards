from ...domain.generation.interfaces.base_algorithm import BaseAlgorithm
from ...infrastructure.generation.algorithms.nsga2 import NSGAII, NSGA2Config


class AlgorithmFactory:
    _registry = {
        "nsga2": lambda config: NSGAII(NSGA2Config(**config)),
    }

    def create(self, config: dict) -> BaseAlgorithm:
        algo_type = config.get("type")

        try:
            algo_ctor = self._registry[algo_type.lower()]
        except KeyError as e:
            raise ValueError(f"Unsupported algorithm: {algo_type}") from e

        return algo_ctor(config)
