from pathlib import Path

from ....domain.generation.entities.data_model import DataModel
from ....domain.generation.interfaces.base_repository import BaseParetoDataRepository
from ...factories.algorithm import AlgorithmFactory
from ...factories.optimizer import OptimizerFactory
from ...factories.problem import ProblemFactory
from .generate_biobj_data_command import GenerateBiobjDataCommand


class GenerateBiobjDataCommandHandler:
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
        data_model_repository: BaseParetoDataRepository,
    ):
        """
        Initializes the command handler with necessary factories and an data_model_repository.

        Args:
            problem_factory: Factory to create problem instances.
            algorithm_factory: Factory to create algorithm instances.
            optimizer_factory: Factory to create optimizer runner instances.
            data_model_repository: Component responsible for saving and loading Pareto data.
        """
        self._problem_factory = problem_factory
        self._algorithm_factory = algorithm_factory
        self._optimizer_factory = optimizer_factory
        self._data_model_repository = data_model_repository

    def execute(self, command: GenerateBiobjDataCommand) -> Path:
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

        # Build the DataModel from the optimization results and original configurations
        data = DataModel(
            name=algorithm_config.get("type", "UnknownAlgorithm"),
            pareto_set=result.pareto_set,
            pareto_front=result.pareto_front,
            historical_solutions=result.all_historical_solutions,
            historical_objectives=result.all_historical_objectives,
            metadata={
                "algorithm": algorithm_config,
                "optimizer": optimizer_config,
                "problem": problem_config,
            },
        )

        # Save the results using the data model repository and return the saved path
        return self._data_model_repository.save(data=data)
