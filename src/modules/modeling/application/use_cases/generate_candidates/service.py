from pydantic import BaseModel, Field

from ....domain.enums.estimator_type import EstimatorTypeEnum


class InverseEstimatorCandidate(BaseModel):
    """Represents a specific inverse estimator candidate (type and optional version)."""

    type: EstimatorTypeEnum = Field(
        ...,
        examples=[EstimatorTypeEnum.MDN.value],
    )
    version: int | None = Field(
        ...,
        description="Specific integer version number (e.g., 1). If None, latest is used.",
        examples=[1],
    )


class GenerateCandidatesParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to use for comparison.",
        examples=["dataset"],
    )

    inverse_estimator: InverseEstimatorCandidate = Field(
        ...,
        description="inverse model candidate (type + version) to use for generation",
        examples={
            "type": EstimatorTypeEnum.MDN.value,
            "version": 1,
        },
    )
    forward_estimator_type: EstimatorTypeEnum = Field(
        ...,
        description="Name of the forward model to use for verification",
        examples=[EstimatorTypeEnum.MDN.value],
    )
    target_objective: list[float] = Field(
        ...,
        description="Target point in objective space (y1, y2, ...)",
        examples=[[0.5, 0.8]],
    )
    distance_tolerance: float = Field(
        ...,
        description="Distance tolerance for selecting candidates.",
        examples=[0.02],
    )
    n_samples: int = Field(
        ...,
        description="Number of samples to draw per inverse estimator.",
        examples=[250],
    )


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


class GenerateCandidatesService:
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

    def execute(self, params: GenerateCandidatesParams) -> dict[str, Any]:
        """
        Coordinates the generation of decision candidates using multiple inverse models.
        """
        self._logger.log_info(
            f"Starting decision generation on '{params.dataset_name}'. "
            f"Inverse: {params.inverse_estimator.type.value}"
            f"Forward: {params.forward_estimator_type.value}"
        )

        # 1. Load context: dataset and processed data
        dataset: Dataset = self._data_repository.load(params.dataset_name)
        if not dataset.processed:
            raise ValueError(f"Dataset '{dataset.name}' has no processed data.")

        # 2. Prepare target objective (raw + normalized)
        target_objective_raw, target_objective_norm = self._prepare_target(
            params.target_objective, dataset.processed
        )
        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 3. Initialize inverse estimators using private helper
        inverse_estimator = self._initialize_estimator(
            candidate=params.inverse_estimator, dataset_name=params.dataset_name
        )

        # 4. Generate candidates
        candidates = inverse_estimator.sample(
            X=target_objective_norm,
            n_samples=params.n_samples,
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
