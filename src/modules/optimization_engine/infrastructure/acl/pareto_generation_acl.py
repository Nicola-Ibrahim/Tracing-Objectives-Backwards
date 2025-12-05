from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain.datasets.entities.dataset import Dataset
from ...domain.datasets.interfaces.base_repository import BaseDatasetRepository


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

    def __init__(self, dataset_repository: BaseDatasetRepository):
        """
        Initializes the ACL with a dependency on the BaseDatasetRepository.

        Args:
            dataset_repository: An instance of the BaseDatasetRepository.
        """
        self._dataset_repo = dataset_repository

    def get_pareto_analysis_data(
        self, data_identifier: Path | str
    ) -> AnalysisResultDTO:
        """
        Retrieves Pareto data and transforms it into a format
        suitable for analysis.

        Args:
            data_identifier: The identifier (e.g., name) of the dataset.

        Returns:
            AnalysisResultDTO: A Data Transfer Object containing the
                               transformed Pareto data for analysis.

        Raises:
            ValueError: If the dataset is not found.
            Exception: Any other exception from the underlying repository.
        """
        # Call the repository from the optimization bounded context
        dataset: Dataset = self._dataset_repo.load(name=str(data_identifier))

        # --- Anti-Corruption / Translation Logic ---
        # Map Dataset aggregate to AnalysisResultDTO.

        metadata = {}
        if dataset.processed and dataset.processed.metadata:
            metadata = dataset.processed.metadata

        transformed_data = AnalysisResultDTO(
            id=data_identifier,
            solutions=dataset.pareto.set,  # Assuming this exists in Pareto value object
            objectives=dataset.pareto.front,
            problem_type=dataset.name,
            original_metadata=metadata,
        )
        print(
            f"ACL: Transformed Dataset ({data_identifier}) to AnalysisResultDTO for analysis."
        )
        return transformed_data
