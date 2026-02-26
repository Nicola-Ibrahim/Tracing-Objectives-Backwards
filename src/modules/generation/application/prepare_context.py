from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.gaussian_process import GaussianProcessRegressor
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


class PrepareContextParams(BaseModel):
    """Configuration for preparing the generation context."""

    dataset_name: str = Field(..., description="Name of the dataset to load")
    k_neighbors: int = Field(
        default=5, description="Number of neighbors for tau computation"
    )
    transforms: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered list of transformer configurations"
    )


class PrepareContextService:
    """
    Orchestrates the offline context preparation for generation:
    1. Loads dataset
    2. Fits data normalizers on the raw data
    3. Transforms data into normalized space
    4. Fits the surrogate model internally
    5. Computes coherence threshold (tau)
    6. Persists the CoherenceContext with explicitly fitted models
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

    def _load_dataset(self, dataset_name: str):
        return self._dataset_repository.load(dataset_name)

    def _fit_and_transform(
        self, data: np.ndarray, transforms: list[BaseTransformer]
    ) -> np.ndarray:
        current_data = data.copy()
        for t in transforms:
            t.fit(current_data)
            current_data = t.transform(current_data)
        return current_data

    def _train_surrogate(self, X_norm: np.ndarray, y_norm: np.ndarray):
        # A simple domain factory explicit to the generation context
        estimator = GaussianProcessRegressor(random_state=42)
        estimator.fit(X_norm, y_norm)
        return estimator

    def execute(
        self,
        params: PrepareContextParams,
    ) -> GenerationContext:
        self._logger.log_info(
            f"Preparing generation context for '{params.dataset_name}'"
        )

        # 1. Load dataset
        dataset = self._load_dataset(params.dataset_name)
        objectives_raw = dataset.objectives
        decisions_raw = dataset.decisions

        # 2. Fit normalizers & transform data sequentially
        decision_transforms = []
        objective_transforms = []

        for t_cfg in params.transforms:
            target_str = t_cfg.get("target")
            transform = self._transformer_factory.create(t_cfg)

            # Set target attribute for persistence/filtering
            if target_str == "decisions":
                target = TransformTarget.DECISIONS
            elif target_str == "objectives":
                target = TransformTarget.OBJECTIVES

            setattr(transform, "target", target)

            if target_str == "decisions":
                decision_transforms.append(transform)
            if target_str == "objectives":
                objective_transforms.append(transform)

        anchors_norm = self._fit_and_transform(decisions_raw, decision_transforms)
        objectives_norm = self._fit_and_transform(objectives_raw, objective_transforms)

        # 4. Train the surrogate
        surrogate_model = self._train_surrogate(anchors_norm, objectives_norm)

        # 5. Compute tau
        nn = NearestNeighbors(n_neighbors=params.k_neighbors + 1)  # +1 to include self
        nn.fit(anchors_norm)
        distances, _ = nn.kneighbors(anchors_norm)

        # distances[:, 1:] ignores the zero-distance to self
        tau = float(np.percentile(distances[:, 1:], 95))
        self._logger.log_info(
            f"Computed tau: {tau:.4f} using k={params.k_neighbors} (95th percentile)"
        )

        # 6. Create and persist Context
        context = GenerationContext(
            dataset_name=params.dataset_name,
            objectives=objectives_raw,  # Storing raw objectives
            anchors_norm=anchors_norm,
            tau=tau,
            transforms=decision_transforms + objective_transforms,
            surrogate_step=surrogate_model,
        )

        self._context_repository.save(context)
        self._logger.log_info("GenerationContext saved successfully.")

        return context
