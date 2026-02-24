from pydantic import BaseModel, Field

from .....modeling.domain.enums.estimator_type import EstimatorTypeEnum


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


class CompareInverseModelCandidatesParams(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Dataset identifier to use for comparison.",
        examples=["dataset"],
    )

    inverse_estimators: list[InverseEstimatorCandidate] = Field(
        ...,
        description="list of inverse model candidates (type + version) to use for generation",
        examples=[
            [
                {"type": EstimatorTypeEnum.MDN.value, "version": 1},
            ]
        ],
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
from ....domain.interfaces.base_visualizer import BaseVisualizer
from .inverse_model_candidates_comparator import InverseModelsCandidatesComparator


class CompareInverseModelCandidatesService:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for Multiple Models:
    - Uses a dedicated inverse generator comparator.
    - Compares generation capability across multiple inverse models (MDN, CVAE, etc).
    """

    def __init__(
        self,
        comparator: InverseModelsCandidatesComparator,
        model_repository: BaseModelArtifactRepository,
        data_repository: BaseDatasetRepository,
        logger: BaseLogger,
        visualizer: BaseVisualizer,
    ):
        self._comparator = comparator
        self._model_repository = model_repository
        self._data_repository = data_repository
        self._logger = logger
        self._visualizer = visualizer

    def execute(self, params: CompareInverseModelCandidatesParams) -> dict[str, Any]:
        """
        Coordinates the generation of decision candidates using multiple inverse models.
        """
        self._logger.log_info(
            f"Starting decision generation on '{params.dataset_name}'. "
            f"Candidates: {[(c.type.value, c.version) for c in params.inverse_estimators]}, "
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
        inverse_estimators = self._initialize_estimators(
            candidates=params.inverse_estimators, dataset_name=params.dataset_name
        )

        # 4. Load forward estimator using repository directly (as requested by user earlier)
        forward_estimator = self._model_repository.get_latest_version(
            estimator_type=params.forward_estimator_type.value,
            mapping_direction="forward",
            dataset_name=params.dataset_name,
        ).estimator

        # 5. Run comparison workflow
        workflow_output = self._comparator.compare(
            inverse_estimators=inverse_estimators,
            forward_estimator=forward_estimator,
            target_objective_norm=target_objective_norm,
            target_objective_raw=target_objective_raw,
            decisions_normalizer=dataset.processed.decisions_normalizer,
            objectives_normalizer=dataset.processed.objectives_normalizer,
            n_samples=params.n_samples,
        )

        # 6. Build visualization payload and generate plots
        results_map = workflow_output["results_map"]
        generator_runs = workflow_output["generator_runs"]

        self._logger.log_info("Generating comparison visualization...")
        self._visualizer.plot(
            data={
                "dataset_name": dataset.name,
                "pareto_front": dataset.pareto.front,
                "pareto_set": dataset.pareto.set,
                "target_objective": target_objective_raw,
                "generators": generator_runs,
            }
        )

        return results_map

    def _initialize_estimators(
        self, candidates: list[InverseEstimatorCandidate], dataset_name: str
    ) -> dict[str, BaseEstimator]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_estimators: dict[str, BaseEstimator] = {}

        for candidate in candidates:
            inverse_type = candidate.type.value
            version = candidate.version
            display_name = (
                f"{inverse_type} (v{version})"
                if version
                else f"{inverse_type} (latest)"
            )

            # Resolve from repository
            artifact = self._model_repository.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            inverse_estimators[display_name] = artifact.estimator

        return inverse_estimators

    def _build_visualization_payload(
        self,
        dataset: Dataset,
        target_objective: np.ndarray,
        generator_runs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the visualization payload for the visualizer.
        """
        return

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
