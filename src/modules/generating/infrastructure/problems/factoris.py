from ...domain.interfaces.base_problem import BaseProblem
from .biobj import BiObjProblemConfig, COCOBiObjectiveProblem, get_coco_problem


class ProblemFactory:
    def create(self, config: dict) -> BaseProblem:
        problem_type = config["type"]
        problem_id = config["id"]

        if problem_type == "biobj":
            coco_problem = get_coco_problem("bbob-biobj", function_indices=problem_id)
            config = BiObjProblemConfig(
                problem_id=problem_id,
                n_var=2,
                n_obj=2,
                n_constr=0,
                xl=coco_problem.lower_bounds,
                xu=coco_problem.upper_bounds,
            )
            return COCOBiObjectiveProblem(coco_problem, config)

        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
