from pathlib import Path

from ....domain.generation.entities.pareto_data import ParetoDataModel
from ....domain.generation.interfaces.base_archiver import BaseParetoArchiver

# Import the factories and archiver directly
from ....infrastructure.algorithms import AlgorithmFactory
from ....infrastructure.optimizers import OptimizerFactory
from ....infrastructure.problems import ProblemFactory
from .generate_pareto_command import GenerateParetoCommand


class GenerateBiobjParetoDataCommandHandler:
    """
    Command handler for generating biobjective Pareto data.
    This handler now directly orchestrates the core generation logic,
    without delegating to a separate ParetoGenerationService.
    """

    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        archiver: BaseParetoArchiver,
    ):
        """
        Initializes the command handler with necessary factories and an archiver.

        Args:
            problem_factory: Factory to create problem instances.
            algorithm_factory: Factory to create algorithm instances.
            optimizer_factory: Factory to create optimizer runner instances.
            archiver: Component responsible for saving and loading Pareto data.
        """
        self._problem_factory = problem_factory
        self._algorithm_factory = algorithm_factory
        self._optimizer_factory = optimizer_factory
        self._archiver = archiver

    def execute(self, command: GenerateParetoCommand) -> Path:
        """
        Executes the command by directly orchestrating the Pareto data generation.

        Args:
            command: The command object containing all necessary configurations.

        Returns:
            Path: The file path where the generated Pareto data is saved.
        """
        problem_config = command.problem_config.model_dump()
        algorithm_config = command.algorithm_config.model_dump()
        optimizer_config = command.optimizer_config.model_dump()

        # Create domain objects using the provided configurations
        problem = self._problem_factory.create(problem_config)
        algorithm = self._algorithm_factory.create(algorithm_config)

        # Create optimizer runner with its dependencies (problem and algorithm)
        optimizer = self._optimizer_factory.create(
            problem=problem,
            algorithm=algorithm,
            config=optimizer_config,
        )

        # Execute the optimization process
        result = optimizer.run()

        # Build the ParetoDataModel from the optimization results and original configurations
        data = ParetoDataModel(
            pareto_set=result.pareto_set,
            pareto_front=result.pareto_front,
            problem_name=algorithm_config.get("type", "UnknownAlgorithm"),
            metadata={
                "algorithm": algorithm_config,
                "optimizer": optimizer_config,
                "problem": problem_config,
            },
        )

        # Save the results using the archiver and return the saved path
        return self._archiver.save(data=data, filename="pareto_data")
