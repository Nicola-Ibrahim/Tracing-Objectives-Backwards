from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...modeling.domain.interfaces.base_transform import (
    BaseTransformer,
    TransformTarget,
)
from ...modeling.infrastructure.factories.transformer import TransformerFactory
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.entities.generation_context import GenerationContext
from ..domain.interfaces.base_context_repository import BaseContextRepository


class TrainContextParams(BaseModel):
    """Configuration for preparing the generation context."""

    dataset_name: str = Field(..., description="Name of the dataset to load")
    k_neighbors: int = Field(
        default=5, description="Number of neighbors for tau computation"
    )
    transforms: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered list of transformer configurations"
    )


class TrainContextService:
    """
    Orchestrates the offline context preparation for generation:
    1. Loads dataset
    2. Fits data normalizers on the raw data
    3. Transforms data into normalized space
    4. Fits the surrogate model internally
    5. Computes coherence threshold (tau)
    6. Builds the mesh triangulation of the objective space
    7. Persists the CoherenceContext with explicitly fitted models
    """

    def __init__(
        self,
        dataset_repository: BaseDatasetRepository,
        context_repository: BaseContextRepository,
        logger: BaseLogger,
        transformer_factory: TransformerFactory | None = None,
    ):
        self._dataset_repository = dataset_repository
        self._context_repository = context_repository
        self._logger = logger
        self._transformer_factory = transformer_factory or TransformerFactory()

    def _fit_and_transform(
        self, data: np.ndarray, transforms: list[tuple[BaseTransformer, Any]]
    ) -> np.ndarray:
        current_data = data.copy()
        for t, _ in transforms:
            t.fit(current_data)
            current_data = t.transform(current_data)
        return current_data

    def _train_surrogate(self, X: np.ndarray, y: np.ndarray):
        from ...modeling.infrastructure.estimators.deterministic.rbf import (
            RBFEstimator,
            RBFEstimatorParams,
        )

        # Use the formal RBFEstimator wrapper with default params for now
        # neighbors=10 is the default in RBFEstimatorParams
        params = RBFEstimatorParams(n_neighbors=10)
        estimator = RBFEstimator(params)
        estimator.fit(X, y)
        return estimator

    def execute(self, params: TrainContextParams) -> GenerationContext:
        self._logger.log_info(
            f"Preparing generation context for '{params.dataset_name}'"
        )

        # Fitting the transforms
        dataset = self._dataset_repository.load(params.dataset_name)
        objectives_raw = dataset.objectives
        decisions_raw = dataset.decisions

        decision_transforms = []
        objective_transforms = []

        for t_cfg in params.transforms:
            target_type = t_cfg.get("target")
            transform = self._transformer_factory.create(t_cfg)

            if target_type == "decisions":
                decision_transforms.append((transform, TransformTarget.DECISIONS))
            elif target_type == "objectives":
                objective_transforms.append((transform, TransformTarget.OBJECTIVES))

        decisions_norm = self._fit_and_transform(decisions_raw, decision_transforms)
        objectives_norm = self._fit_and_transform(objectives_raw, objective_transforms)

        # Training the Surrogate Model (The actual X -> Y mapping)
        surrogate_estimator = self._train_surrogate(decisions_norm, objectives_norm)

        # Computing the Coherence Threshold (tau)
        nn = NearestNeighbors(n_neighbors=params.k_neighbors + 1)  # +1 to include self
        nn.fit(decisions_norm)
        distances, _ = nn.kneighbors(decisions_norm)

        # distances[:, 1:] ignores the zero-distance to self
        tau = float(np.percentile(distances[:, 1:], 95))
        self._logger.log_info(
            f"Computed tau: {tau:.4f} using k={params.k_neighbors} (95th percentile)"
        )

        # Create the mesh for the objective space
        mesh = Delaunay(objectives_norm)

        # Build the spatial index for the objective space (for out-of-mesh lookups)
        objective_knn = NearestNeighbors(n_neighbors=1)
        objective_knn.fit(objectives_norm)

        # Creating the GenerationContext
        context = GenerationContext(
            dataset_name=params.dataset_name,
            space_points=objectives_norm,
            decision_vertices=decisions_norm,
            tau=tau,
            mesh=mesh,
            objective_knn=objective_knn,
            transforms=decision_transforms + objective_transforms,
            surrogate_estimator=surrogate_estimator,
            is_trained=True,
        )

        self._context_repository.save(context)
        self._logger.log_info("GenerationContext saved successfully.")

        return context
