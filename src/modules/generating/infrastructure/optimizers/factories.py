from ...domain.interfaces.base_algorithm import BaseAlgorithm
from ...domain.interfaces.base_optimizer import BaseOptimizer
from ...domain.interfaces.base_problem import BaseProblem
from .minimizer import Minimizer, MinimizerConfig


class OptimizerFactory:
    def create(
        self,
        problem: BaseProblem,
        algorithm: BaseAlgorithm,
        config: dict,
    ) -> BaseOptimizer:
        # Implementation creates optimizer with its dependencies

        opt_type = config["type"]
        if opt_type == "minimizer":
            minimizer_config = MinimizerConfig(
                generations=16,
                seed=42,
                save_history=False,  # Avoids deepcopy of problem object
                verbose=False,
                pf=True,
            )

            return Minimizer(
                problem=problem, algorithm=algorithm, config=minimizer_config
            )

        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
