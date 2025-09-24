from pathlib import Path

from ....domain.common.interfaces.base_logger import BaseLogger
from ....domain.datasets.entities.generated_dataset import GeneratedDataset
from ....domain.datasets.interfaces.base_repository import BaseDatasetRepository
from ....domain.datasets.value_objects.pareto import (
    Pareto,
)
from ...factories.algorithm import AlgorithmFactory
from ...factories.optimizer import OptimizerFactory
from ...factories.problem import ProblemFactory
from .generate_dataset_command import GenerateDatasetCommand


class GenerateDatasetCommandHandler:
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
        data_model_repository: BaseDatasetRepository,
        logger: BaseLogger,
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
        self._logger = logger

    def execute(self, command: GenerateDatasetCommand) -> Path:
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
        run_data = optimizer.run()

        self._logger.log_info("Optimization run completed.")

        self._logger.log_info(
            f"Found {run_data.pareto_set.shape[0]} Pareto-optimal solutions."
        )
        self._logger.log_info(
            f"Historical Pareto set contains {run_data.historical_solutions.shape[0]} solutions."
        )

        pareto = Pareto(
            set=run_data.pareto_set,
            front=run_data.pareto_front,
        )

        # Build the GeneratedDataset from the optimization results and original configurations
        generated_dataset = GeneratedDataset(
            name="dataset",
            X=run_data.historical_solutions,
            y=run_data.historical_objectives,
            pareto=pareto,
        )

        # Save the results using the data model repository and return the saved path
        saved_path = self._data_model_repository.save(data=generated_dataset)
        self._logger.log_info(f"Pareto data saved to: {saved_path}")
