from ...domain.datasets.interfaces.base_problem import BaseProblem
from ...infrastructure.generation.problems.biobj import (
    BiObjProblemConfig,
    COCOBiObjectiveProblem,
)


class ProblemFactory:
    _registry = {
        "biobj": lambda config: COCOBiObjectiveProblem(
            BiObjProblemConfig(**config, n_var=2, n_obj=2, n_constr=0)
        ),
    }

    def create(self, config: dict) -> BaseProblem:
        problem_type = config.get("type")

        try:
            problem_ctor = self._registry[problem_type]

        except KeyError as e:
            raise ValueError(f"Unsupported problem type: {problem_type}") from e

        return problem_ctor(config)
