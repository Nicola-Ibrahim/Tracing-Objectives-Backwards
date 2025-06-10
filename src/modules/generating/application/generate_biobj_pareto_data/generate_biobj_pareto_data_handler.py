from pathlib import Path

from ....shared.adapters.archivers.base import BaseParetoArchiver
from ...adapters.algorithms import AlgorithmFactory
from ...adapters.optimizers import OptimizerFactory
from ...adapters.problems import ProblemFactory
from ...domain.entities.pareto_data import ParetoDataModel
from .generate_pareto_command import GenerateParetoCommand


class GenerateBiobjParetoDataHandler:
    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        archiver: BaseParetoArchiver,
    ):
        self._problem_factory = problem_factory
        self._algorithm_factory = algorithm_factory
        self._optimizer_factory = optimizer_factory
        self._archiver = archiver

    def execute(self, command: GenerateParetoCommand) -> Path:
        # Create domain objects
        problem = self._problem_factory.create(command.problem_config.model_dump())
        algorithm = self._algorithm_factory.create(
            command.algorithm_config.model_dump()
        )

        # Create optimizer runner with dependencies
        optimizer = self._optimizer_factory.create(
            problem=problem,
            algorithm=algorithm,
            config=command.optimizer_config.model_dump(),
        )

        # Execute optimization
        result = optimizer.run()

        # Build data model
        data = ParetoDataModel(
            pareto_set=result.pareto_set,
            pareto_front=result.pareto_front,
            problem_name=problem.name,
            metadata={
                "algorithm": command.algorithm_config.name,
                "optimizer": command.optimizer_config.model_dump(),
                "problem": problem.config.model_dump(),
            },
        )

        # Save results
        return self._archiver.save(data)
