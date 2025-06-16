from pathlib import Path
from typing import Any

from ...domain.entities.pareto_data import ParetoDataModel
from ...domain.interfaces.base_archiver import BaseParetoArchiver
from ...infrastructure.algorithms import AlgorithmFactory
from ...infrastructure.optimizers import OptimizerFactory
from ...infrastructure.problems import ProblemFactory


class ParetoGenerationService:
    """
    Service responsible for orchestrating the generation and retrieval of Pareto sets and fronts.
    It encapsulates the workflow of creating problems, algorithms, running optimization,
    and archiving/retrieving the results.
    """

    def __init__(
        self,
        problem_factory: ProblemFactory,
        algorithm_factory: AlgorithmFactory,
        optimizer_factory: OptimizerFactory,
        archiver: BaseParetoArchiver,
    ):
        """
        Initializes the ParetoGenerationService with necessary factories and an archiver.

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

    def generate_pareto_data(
        self,
        problem_config: dict[str, Any],
        algorithm_config: dict[str, Any],
        optimizer_config: dict[str, Any],
    ) -> Path:
        """
        Generates Pareto data based on provided configuration dictionaries.

        This method orchestrates the creation of a problem, an algorithm, and an optimizer,
        executes the optimization, builds a ParetoDataModel from the results, and
        then archives this data.

        Args:
            problem_config: Configuration for the optimization problem.
            algorithm_config: Configuration for the optimization algorithm.
            optimizer_config: Configuration for the optimizer runner.

        Returns:
            Path: The file path where the generated Pareto data is saved.
        """
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
            # Use the algorithm type from config for problem_name as in original code,
            # or consider a more descriptive name if available in problem_config
            problem_name=algorithm_config.get("type", "UnknownAlgorithm"),
            metadata={
                "algorithm": algorithm_config,
                "optimizer": optimizer_config,
                "problem": problem_config,
            },
        )

        # Save the results using the archiver and return the saved path
        return self._archiver.save(data)

    def retrieve_pareto_data(
        self,
        data_identifier: Path | str,
    ) -> ParetoDataModel:
        """
        Retrieves previously archived Pareto data.

        This method acts as an access point for other bounded contexts (e.g., an
        analyzing bounded context via an ACL) to retrieve stored Pareto data
        without needing to know the underlying storage mechanism.

        Args:
            data_identifier: An identifier for the data to retrieve, typically
                             the path returned by `generate_pareto_data`.

        Returns:
            ParetoDataModel: The retrieved Pareto data model.

        Raises:
            FileNotFoundError: If the data specified by `data_identifier` is not found.
            Exception: Any other exception raised by the underlying archiver during loading.
        """
        # The archiver is responsible for knowing how to load the data given an identifier.
        # It's assumed that the BaseParetoArchiver will have a 'load' method that
        # takes a path or similar identifier and returns a ParetoDataModel.
        return self._archiver.load(data_identifier)
