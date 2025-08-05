from ...domain.generation.interfaces.base_problem import BaseProblem
from .biobj import BiObjProblemConfig, COCOBiObjectiveProblem


class ProblemFactory:
    def create(self, config: dict) -> BaseProblem:
        problem_type = config["type"]
        problem_id = config["id"]

        if problem_type == "biobj":
            config = BiObjProblemConfig(
                problem_id=problem_id,
                n_var=2,
                n_obj=2,
                n_constr=0,
            )
            return COCOBiObjectiveProblem(config)

        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
