from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.entities.trained_pipeline import TrainedPipeline
from ...domain.enums.estimator_type import EstimatorTypeEnum
from ...domain.interfaces.base_repository import BaseTrainedPipelineRepository
from .models import InverseEstimatorCandidate


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


class GenerateCandidatesService:
    """
    Generate Design Candidates (X) for a requested Objective (Y).

    Unified Handler for Multiple Models:
    - Uses a dedicated inverse generator comparator.
    - Compares generation capability across multiple inverse models (MDN, CVAE, etc).
    """

    def __init__(
        self,
        model_repository: BaseTrainedPipelineRepository,
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
            f"Inverse: {params.inverse_estimator.type.value} "
            f"Forward: {params.forward_estimator_type.value}"
        )

        # 2. Get pipeline
        pipeline = self._model_repository.get_version_by_number(
            estimator_type=params.inverse_estimator.type.value,
            version=params.inverse_estimator.version,
            mapping_direction="inverse",
            dataset_name=params.dataset_name,
        )

        # 3. Prepare target objective (raw + normalized)
        target_objective_raw, target_objective_norm = self._prepare_target(
            params.target_objective, pipeline
        )
        self._logger.log_info(
            f"Target Objective (Raw): {target_objective_raw.tolist()}"
        )

        # 4. Generate candidates
        inverse_estimator = pipeline.model.fitted
        candidates_norm = inverse_estimator.sample(
            X=target_objective_norm,
            n_samples=params.n_samples,
        )

        # candidates_norm shape is (n_samples, n_decisions) assuming sample() returns that.
        # old code traversed candidates.shape[1] so it might be (n_decisions, n_samples)?
        # Actually usually it's (n_samples, n_decisions) but the loop said `candidates[:, i]`.
        # Let's ensure shape is correct for inverse_transform.
        # inverse_transform needs (n_samples, n_features).
        if (
            len(candidates_norm.shape) == 2
            and candidates_norm.shape[1] == params.n_samples
        ):
            # It's (n_decisions, n_samples)
            candidates_to_transform = candidates_norm.T
        elif len(candidates_norm.shape) == 3:  # (batch, n_samples, dim)
            candidates_to_transform = candidates_norm[0]
        else:
            candidates_to_transform = candidates_norm

        dec_transforms = pipeline.get_decisions_transforms()
        candidates_raw = candidates_to_transform.copy()

        if dec_transforms:
            for t in reversed(dec_transforms):
                candidates_raw = t.inverse_transform(candidates_raw)
        else:
            raise ValueError(
                "Pipeline has no transforms for decision space normalization."
            )

        # restore shape if necessary
        if (
            len(candidates_norm.shape) == 2
            and candidates_norm.shape[1] == params.n_samples
        ):
            candidates = candidates_raw.T
        elif len(candidates_norm.shape) == 3:
            candidates = candidates_raw.reshape(candidates_norm.shape)
        else:
            candidates = candidates_raw

        # Log
        # Using old shape assumptions for logging
        if len(candidates.shape) == 2 and candidates.shape[1] == params.n_samples:
            self._logger.log_info(f"Generated {candidates.shape[1]} candidates.")
            for i in range(candidates.shape[1]):
                print(f"Candidate {i}: {candidates[:, i].tolist()}")
        else:
            self._logger.log_info(f"Generated {len(candidates)} candidates.")
            for i in range(len(candidates)):
                print(f"Candidate {i}: {candidates[i].tolist()}")

        return {"candidates": candidates}

    def _prepare_target(
        self,
        target_objective: list,
        pipeline: TrainedPipeline,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares raw and normalized target objectives.
        """
        target_objective_raw = np.array(target_objective, dtype=float).reshape(1, -1)

        target_objective_norm = target_objective_raw.copy()
        obj_transforms = pipeline.get_objectives_transforms()

        if obj_transforms:
            for t in obj_transforms:
                target_objective_norm = t.transform(target_objective_norm)
        else:
            raise ValueError(
                "No transforms found in pipeline and no processed data fallback available."
            )

        return target_objective_raw, target_objective_norm
