from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ....dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ....modeling.domain.entities.trained_pipeline import TrainedPipeline
from ....modeling.domain.enums.estimator_type import EstimatorTypeEnum
from ....modeling.domain.interfaces.base_estimator import ProbabilisticEstimator
from ....modeling.domain.interfaces.base_repository import BaseTrainedPipelineRepository
from ....modeling.domain.services.preprocessing_service import PreprocessingService
from ....shared.domain.interfaces.base_logger import BaseLogger
from ...domain.aggregates.diagnostic_result import DiagnosticResult
from ...domain.entities.accuracy_lens import AccuracyLens
from ...domain.entities.reliability_lens import ReliabilityLens
from ...domain.interfaces.base_diagnostic_repository import (
    BaseDiagnosticRepository,
)
from ...domain.services.generative_distribution_auditor import (
    GenerativeDistributionAuditor,
)
from ...domain.services.spatial_candidate_auditor import (
    SpatialCandidateAuditor,
)
from ...domain.value_objects.estimator import Estimator
from .models import InverseEstimatorCandidate


class DiagnoseInverseModelsParams(BaseModel):
    """
    Command for the full evaluation suite including
    Objective-Space Accuracy and Decision-Space Reliability.
    Supports comparing multiple inverse model candidates.
    """

    dataset_name: str = Field(..., examples=["cocoex_f5"])

    inverse_estimator_candidates: list[InverseEstimatorCandidate] = Field(
        ...,
        description="List of model candidates to compare.",
        examples=[[{"type": EstimatorTypeEnum.MDN.value, "version": 1}]],
    )

    forward_estimator_type: EstimatorTypeEnum = Field(
        ..., examples=[EstimatorTypeEnum.COCO]
    )

    num_samples: int = Field(default=200, description="K candidates per target")
    random_state: int = 42

    scale_method: Literal["sd", "mad", "iqr"] = Field(
        default="sd", description="sd | mad | iqr"
    )


class DiagnoseInverseModelsService:
    """
    Orchestrator for the Thesis-aligned evaluation suite.
    Computes diagnostics and persists them via the Diagnostic Repository.
    """

    def __init__(
        self,
        model_repository: BaseTrainedPipelineRepository,
        data_repository: BaseDatasetRepository,
        diagnostic_repository: BaseDiagnosticRepository,
        logger: BaseLogger,
        preprocessing_service: PreprocessingService,
    ):
        self._model_repository = model_repository
        self._data_repo = data_repository
        self._diag_repo = diagnostic_repository
        self._logger = logger
        self._preprocessing_service = preprocessing_service

    def execute(self, params: DiagnoseInverseModelsParams) -> dict[str, int]:
        self._logger.log_info(
            f"Starting Diagnostic Compute for: "
            f"{[(c.type.value, c.version) for c in params.inverse_estimator_candidates]}"
        )

        dataset = self._data_repo.load(params.dataset_name)

        forward_pipeline = self._model_repository.get_latest_version(
            params.forward_estimator_type.value, "forward", params.dataset_name
        )

        inverse_pipelines = self._initialize_estimators(
            inverse_estimator_candidates=params.inverse_estimator_candidates,
            dataset_name=params.dataset_name,
        )

        X_raw = dataset.objectives
        y_raw = dataset.decisions

        for version, inverse_pipeline in inverse_pipelines:
            # 1. Split according to inverse pipeline
            split_step, X_train_raw, X_test_raw, y_train_raw, y_test_raw = (
                self._preprocessing_service.split(X_raw, y_raw, inverse_pipeline.split)
            )

            # 2. Transform the test split to normalized space
            X_test_norm = X_test_raw.copy()
            for t in inverse_pipeline.get_objectives_transforms():
                X_test_norm = t.transform(X_test_norm)

            y_test_norm = y_test_raw.copy()
            for t in inverse_pipeline.get_decisions_transforms():
                y_test_norm = t.transform(y_test_norm)

            # Sampling (candidates in X_norm space)
            # Notice X=objectives, y=decisions for inverse mapping!
            samples_X_norm = inverse_pipeline.model.fitted.sample(
                X_test_norm, n_samples=params.num_samples
            )

            if samples_X_norm.ndim == 2:
                samples_X_norm = samples_X_norm[:, np.newaxis, :]

            n_test, k_samples, x_dim = samples_X_norm.shape

            # Reshape the X-normalized-space samples to a flat array
            samples_X_norm_flat = samples_X_norm.reshape(-1, x_dim)

            # Denormalize the X-normalized-space samples to physical space
            samples_X_phys_flat = samples_X_norm_flat.copy()
            for t in reversed(inverse_pipeline.get_decisions_transforms()):
                samples_X_phys_flat = t.inverse_transform(samples_X_phys_flat)

            # Forward simulate the X-space samples
            # First normalize using Forward model's decision normalizers
            fwd_samples_X_norm_flat = samples_X_phys_flat.copy()
            for t in forward_pipeline.get_decisions_transforms():
                fwd_samples_X_norm_flat = t.transform(fwd_samples_X_norm_flat)

            # Forward model prediction
            pred_obj_norm_fwd_flat = forward_pipeline.model.fitted.predict(
                fwd_samples_X_norm_flat
            )

            # Map back to physical objective space
            pred_obj_phys_flat = pred_obj_norm_fwd_flat.copy()
            for t in reversed(forward_pipeline.get_objectives_transforms()):
                pred_obj_phys_flat = t.inverse_transform(pred_obj_phys_flat)

            # Now we must map it back to the inverse model's NORMALIZED objective space for fair comparison
            # against `X_test_norm` (which was the target).
            pred_obj_norm_inv_flat = pred_obj_phys_flat.copy()
            for t in inverse_pipeline.get_objectives_transforms():
                pred_obj_norm_inv_flat = t.transform(pred_obj_norm_inv_flat)

            # Reshape the y-space predictions to match the test set shape
            y_pred = pred_obj_norm_inv_flat.reshape(n_test, k_samples, -1)

            # Accuracy Domain (Objective Space - Standardized Lens)
            y_target = X_test_norm

            # Train objectives in normalized space for the auditor
            y_train_norm = X_train_raw.copy()
            for t in inverse_pipeline.get_objectives_transforms():
                y_train_norm = t.transform(y_train_norm)

            spatial_audit = SpatialCandidateAuditor.audit(
                training_objectives=y_train_norm,
                candidates=y_pred,
                reference=y_target,
                distance="euclidean",
            )

            # Reliability Domain (Decision Space - Probabilistic Lens)
            x_truth = y_test_norm
            generative_audit = GenerativeDistributionAuditor.audit(
                samples=samples_X_norm,
                truth=x_truth,
            )

            # Create Diagnostic Result Aggregate
            result = DiagnosticResult.create(
                estimator=Estimator(
                    type=inverse_pipeline.model.fitted.type,
                    version=version,
                    mapping_direction="inverse",
                ),
                dataset_name=params.dataset_name,
                num_samples=params.num_samples,
                scale_method=params.scale_method,
                accuracy=AccuracyLens(
                    discrepancy_scores=spatial_audit.discrepancy_scores,
                    best_shot_scores=spatial_audit.best_shot_scores,
                    rank_indices=spatial_audit.rank_indices,
                    systematic_bias=spatial_audit.bias,
                    cloud_dispersion=spatial_audit.dispersion,
                    summary=spatial_audit.summary,
                ),
                reliability=ReliabilityLens(
                    pit_values=generative_audit.pit_values,
                    calibration_error=generative_audit.calibration_error,
                    crps=generative_audit.crps,
                    diversity=generative_audit.diversity,
                    interval_width=generative_audit.interval_width,
                    summary=generative_audit.summary,
                    calibration_curve=generative_audit.calibration_curve,
                ),
            )

            # Persist Result
            self._diag_repo.save(result)

    def _initialize_estimators(
        self,
        inverse_estimator_candidates: list[InverseEstimatorCandidate],
        dataset_name: str,
    ) -> list[tuple[int, TrainedPipeline]]:
        """
        Initializes inverse estimator instances from the repository.
        """
        inverse_pipelines: list[tuple[int, TrainedPipeline]] = []

        for candidate in inverse_estimator_candidates:
            inverse_type = candidate.type.value
            version = candidate.version

            # Resolve from repository
            pipeline = self._model_repository.get_version_by_number(
                estimator_type=inverse_type,
                version=version,
                mapping_direction="inverse",
                dataset_name=dataset_name,
            )

            if not isinstance(pipeline.model.fitted, ProbabilisticEstimator):
                self._logger.log_info(
                    f"Estimator {inverse_type} (v{version}) is not probabilistic. Statistical metrics might be unreliable."
                )

            inverse_pipelines.append((version, pipeline))

        return inverse_pipelines
