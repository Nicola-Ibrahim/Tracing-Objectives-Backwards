from pathlib import Path

from ....domain.services.pareto_generation_service import ParetoGenerationService
from .generate_pareto_command import GenerateParetoCommand


class GenerateBiobjParetoDataCommandHandler:
    """
    Command handler for generating biobjective Pareto data.
    This handler now delegates the core generation logic to ParetoGenerationService,
    acting as an orchestrator for a specific command.
    """

    def __init__(self, pareto_generation_service: ParetoGenerationService):
        """
        Initializes the command handler with a ParetoGenerationService instance.

        Args:
            pareto_generation_service: The service responsible for generating Pareto data.
        """
        self._pareto_generation_service = pareto_generation_service

    def execute(self, command: GenerateParetoCommand) -> Path:
        """
        Executes the command by invoking the ParetoGenerationService.

        Args:
            command: The command object containing all necessary configurations.

        Returns:
            Path: The file path where the generated Pareto data is saved.
        """
        # The handler extracts the raw configuration data from the command
        # and passes it directly to the service.
        return self._pareto_generation_service.generate_pareto_data(
            problem_config=command.problem_config.model_dump(),
            algorithm_config=command.algorithm_config.model_dump(),
            optimizer_config=command.optimizer_config.model_dump(),
        )
