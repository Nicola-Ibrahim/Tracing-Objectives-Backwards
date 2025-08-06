from ...domain.generation.interfaces.base_algorithm import BaseAlgorithm
from .nsga2 import NSGAII, NSGA2Config


class AlgorithmFactory:
    def create(self, config: dict) -> BaseAlgorithm:
        algo_type = config["type"]
        if algo_type.lower() == "nsga2":
            return NSGAII(NSGA2Config(**config))
        else:
            raise ValueError(f"Unsupported algorithm: {algo_type}")
