from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ....dataset.domain.entities.dataset import Dataset
from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modeling.domain.entities.trained_pipeline import TrainedPipeline
from ....modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ....modeling.domain.interfaces.base_repository import (
    BaseTrainedPipelineRepository,
)
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.interfaces.base_visualizer import BaseVisualizer
from .inverse_model_candidates_comparator import InverseModelsCandidatesComparator
from .models import InverseEstimatorCandidate


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
        model_repository: BaseTrainedPipelineRepository,
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

        dataset: Dataset = self._data_repository.load(params.dataset_name)

        target_objective_raw = np.array(params.target_objective, dtype=float).reshape(
            1, -1
        )
        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        inverse_pipelines = self._initialize_estimators(
            candidates=params.inverse_estimators, dataset_name=params.dataset_name
        )

        target_objective_norm_dict = {}
        for name, pipeline in inverse_pipelines.items():
            target_norm = target_objective_raw.copy()
            for t in pipeline.get_objectives_transforms():
                target_norm = t.transform(target_norm)
            target_objective_norm_dict[name] = target_norm

        forward_pipeline = self._model_repository.get_latest_version(
            estimator_type=params.forward_estimator_type.value,
            mapping_direction="forward",
            dataset_name=params.dataset_name,
        )

        workflow_output = self._comparator.compare(
            inverse_pipelines=inverse_pipelines,
            forward_pipeline=forward_pipeline,
            target_objective_norm=target_objective_norm_dict,
            target_objective_raw=target_objective_raw,
            n_samples=params.n_samples,
        )

        results_map = workflow_output["results_map"]
        generator_runs = workflow_output["generator_runs"]

        self._logger.log_info("Generating comparison visualization...")
        self._visualizer.plot(
            data={
                "dataset_name": dataset.name,
                "pareto_front": dataset.pareto.front if dataset.pareto else [],
                "pareto_set": dataset.pareto.set if dataset.pareto else [],
                "target_objective": target_objective_raw,
                "generators": generator_runs,
            }
        )

        return results_map

    def _initialize_estimators(
        self, candidates: list[InverseEstimatorCandidate], dataset_name: str
    ) -> dict[str, TrainedPipeline]:
        """
        Initializes inverse estimator pipelines from the repository.
        """
        inverse_pipelines: dict[str, TrainedPipeline] = {}

        for candidate in candidates:
            inverse_type = candidate.type.value
            version = candidate.version
            display_name = (
                f"{inverse_type} (v{version})"
                if version
                else f"{inverse_type} (latest)"
            )

            pipeline = self._model_repository.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            inverse_pipelines[display_name] = pipeline

        return inverse_pipelines

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
