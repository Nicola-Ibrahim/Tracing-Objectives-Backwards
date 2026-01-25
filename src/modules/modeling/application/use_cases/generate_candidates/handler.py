from typing import Any

import numpy as np

from .....dataset.domain.entities.dataset import Dataset
from .....dataset.domain.entities.processed_data import ProcessedData
from .....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from .....modeling.domain.interfaces.base_estimator import BaseEstimator
from .....modeling.domain.interfaces.base_repository import (
    BaseModelArtifactRepository,
)
from .....shared.domain.interfaces.base_logger import BaseLogger
from .command import GenerateCandidatesCommand, InverseEstimatorCandidate


class GenerateCandidatesCommandHandler:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for Multiple Models:
    - Uses a dedicated inverse generator comparator.
    - Compares generation capability across multiple inverse models (MDN, CVAE, etc).
    """

    def __init__(
        self,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
    ):
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._logger = logger

    def execute(self, command: GenerateCandidatesCommand) -> dict[str, Any]:
        """
        Coordinates the generation of decision candidates using multiple inverse models.
        """
        self._logger.log_info(
            f"Starting decision generation on '{command.dataset_name}'. "
            f"Inverse: {command.inverse_estimator.type.value}"
            f"Forward: {command.forward_estimator_type.value}"
        )

        # 1. Load context: dataset and processed data
        dataset: Dataset = self._data_repository.load(command.dataset_name)
        if not dataset.processed:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2. Prepare target objective (raw + normalized)
        target_objective_raw, target_objective_norm = self._prepare_target(
            command.target_objective, dataset.processed
        )
        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 3. Initialize inverse estimators using private helper
        inverse_estimator = self._initialize_estimator(
            candidate=command.inverse_estimator, dataset_name=command.dataset_name
        )

        # 4. Generate candidates
        candidates = inverse_estimator.sample(
            X=target_objective_norm,
            n_samples=command.n_samples,
        )  

        self._logger.log_info(f"Generated {candidates.shape[1]} candidates.")

        # log the candidate in good shape in terminal and row by row
        for i in range(candidates.shape[1]):
            print(f"Candidate {i}: {candidates[:, i].tolist()}")

        return {"candidates": candidates}

    def _initialize_estimator(
        self, candidate: InverseEstimatorCandidate, dataset_name: str
    ) -> BaseEstimator:
        """
        Initializes inverse estimator instances from the repository.
        """

        inverse_type = candidate.type.value
        version = candidate.version

        # Resolve from repository
        artifact = self._model_repository.get_version_by_number(
            estimator_type=inverse_type,
            version=version,
            mapping_direction="inverse",
            dataset_name=dataset_name,
        )

        return artifact.estimator

    def _prepare_target(
        self, target_objective: list, processed_data: ProcessedData
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares raw and normalized target objectives.
        """
        target_objective_raw = np.array(target_objective, dtype=float).reshape(1, -1)
        target_objective_norm = processed_data.objectives_normalizer.transform(
            target_objective_raw
        )
        return target_objective_raw, target_objective_norm
