from ...domain.generation.interfaces.base_problem import BaseProblem
from ...infrastructure.problems.biobj import BiObjProblemConfig, COCOBiObjectiveProblem


class ProblemFactory:
    _registry = {
        "biobj": lambda config: COCOBiObjectiveProblem(
            BiObjProblemConfig(**config, n_var=2, n_obj=2, n_constr=0)
        ),
    }

    def create(self, config: dict) -> BaseProblem:
        problem_type = config["type"]

        if problem_type not in self._registry:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        return self._registry[problem_type](config)
