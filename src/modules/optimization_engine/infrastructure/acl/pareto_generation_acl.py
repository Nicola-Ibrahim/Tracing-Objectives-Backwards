from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain.generation.entities.data_model import (
    DataModel,
)
from ....optimization_engine.domain.services.pareto_generation_service import (
    ParetoGenerationService,
)


@dataclass
class AnalysisResultDTO:
    """
    Data Transfer Object (DTO) representing a simplified view of Pareto data
    tailored for the Analyzing Bounded Context.
    """

    id: Path | str
    solutions: list[list[float]]
    objectives: list[list[float]]
    problem_type: str
    original_metadata: dict[str, Any]


class GenerationDataACL:
    """
    Anti-Corruption Layer (ACL) for accessing Pareto generation data
    from the 'generation' bounded context within the 'analyzing' bounded context.

    This layer translates concepts and data structures from the generation context
    into a format suitable for the analyzing context, preventing direct coupling
    and shielding the analyzing context from changes in the generation context's
    internal domain model or infrastructure.
    """

    def __init__(self, pareto_generation_service: ParetoGenerationService):
        """
        Initializes the ACL with a dependency on the ParetoGenerationService.

        Args:
            pareto_generation_service: An instance of the ParetoGenerationService
                                       from the generation bounded context.
        """
        self._pareto_service = pareto_generation_service

    def get_pareto_analysis_data(
        self, data_identifier: Path | str
    ) -> AnalysisResultDTO:
        """
        Retrieves Pareto data and transforms it into a format
        suitable for analysis.

        Args:
            data_identifier: The identifier (e.g., file path) of the Pareto data.

        Returns:
            AnalysisResultDTO: A Data Transfer Object containing the
                               transformed Pareto data for analysis.

        Raises:
            FileNotFoundError: If the data specified by `data_identifier` is not found.
            Exception: Any other exception from the underlying service.
        """
        # Call the service from the optimization bounded context
        raw_pareto_data: DataModel = self._pareto_service.retrieve_pareto_data(
            data_identifier
        )

        # --- Anti-Corruption / Translation Logic ---
        # This is where the core ACL work happens.
        # We map the DataModel from the 'optimization' context
        # to the AnalysisResultDTO specific to the 'analyzing' context.
        # If the 'analyzing' context needed different names for fields,
        # or aggregated data, this is where that transformation would occur.

        transformed_data = AnalysisResultDTO(
            id=data_identifier,
            solutions=raw_pareto_data.pareto_set,
            objectives=raw_pareto_data.pareto_front,
            problem_type=raw_pareto_data.problem_name,
            original_metadata=raw_pareto_data.metadata,
        )
        print(
            f"ACL: Transformed DataModel ({data_identifier}) to AnalysisResultDTO for analysis."
        )
        return transformed_data
