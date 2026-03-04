from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...modeling.domain.interfaces.base_transform import BaseTransformer
from ...modeling.infrastructure.factories.transformer import TransformerFactory
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.entities.inverse_mapping_engine import InverseMappingEngine
from ..domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.value_objects.data_split import DataSplit
from ..domain.value_objects.transform_pipeline import TransformPipeline
from ..infrastructure.solvers.factory import SolversFactory


class SolverConfig(BaseModel):
    """Configuration for the inverse solver."""

    type: str = Field(..., description="Solver type (e.g., GBPI)")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Custom solver parameters"
    )


class TrainInverseMappingEngineParams(BaseModel):
    """Configuration for preparing the generation context."""

    dataset_name: str = Field(..., description="Name of the dataset to load")
    solver: SolverConfig = Field(
        ..., description="Inverse solver strategy configuration"
    )
    transforms: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered list of transformer configurations"
    )
    split_ratio: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Ratio of data used for testing"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class TrainInverseMappingEngineService:
    """
    Orchestrates the offline context preparation for generation:
    1. Loads dataset
    2. Splits data into train and test sets
    3. Fits data normalizers on the training data
    4. Transforms data into normalized space
    5. Fits the inverse solver internally
    6. Persists the InverseMappingEngine with versioned hierarchy and value objects
    """

    def __init__(
        self,
        dataset_repository: BaseDatasetRepository,
        inverse_mapping_engine_repository: BaseInverseMappingEngineRepository,
        logger: BaseLogger,
        transformer_factory: TransformerFactory,
        solvers_factory: SolversFactory,
    ):
        self._dataset_repository = dataset_repository
        self._inverse_mapping_engine_repository = inverse_mapping_engine_repository
        self._logger = logger
        self._transformer_factory = transformer_factory
        self._solvers_factory = solvers_factory

    def _fit_and_transform(
        self, data: np.ndarray, transforms: list[tuple[str, BaseTransformer]]
    ) -> np.ndarray:
        current_data = data.copy()
        for _, t in transforms:
            t.fit(current_data)
            current_data = t.transform(current_data)
        return current_data

    def execute(self, params: TrainInverseMappingEngineParams) -> dict:
        import time

        start_time = time.time()
        self._logger.log_info(
            f"Preparing generation context for '{params.dataset_name}'"
        )

        # Loading the dataset
        dataset = self._dataset_repository.load(params.dataset_name)
        objectives_raw = dataset.objectives
        decisions_raw = dataset.decisions

        # Split data into train and test
        total_samples = len(objectives_raw)
        indices = np.arange(total_samples)

        if params.split_ratio > 0.0:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=params.split_ratio,
                random_state=params.random_state,
                shuffle=True,
            )
        else:
            train_indices = indices
            test_indices = np.array([], dtype=int)

        self._logger.log_info(
            f"Split data into {len(train_indices)} training and {len(test_indices)} testing samples."
        )

        # Slice data for training
        X_train = decisions_raw[train_indices]
        y_train = objectives_raw[train_indices]

        # Fitting the transforms (only on training data)
        all_fitted_transforms = []

        for t_cfg in params.transforms:
            target_type = t_cfg.get("target")
            transform = self._transformer_factory.create(t_cfg)
            all_fitted_transforms.append((target_type, transform))

        # We need to transform X_train and y_train for solver training
        # Note: Logic here assumes transforms are applied sequentially per target type
        X_train_norm = X_train.copy()
        for label, t in all_fitted_transforms:
            if label == "decisions":
                t.fit(X_train_norm)
                X_train_norm = t.transform(X_train_norm)

        y_train_norm = y_train.copy()
        for label, t in all_fitted_transforms:
            if label == "objectives":
                t.fit(y_train_norm)
                y_train_norm = t.transform(y_train_norm)

        # Train the solver
        solver_params = params.solver.params.copy()
        if params.solver.type == "GBPI":
            # Supply defaults if missing to ensure successful instantiation
            if "n_neighbors" not in solver_params:
                solver_params["n_neighbors"] = 5
            if "trust_radius" not in solver_params:
                solver_params["trust_radius"] = 0.05
            if "concentration_factor" not in solver_params:
                solver_params["concentration_factor"] = 10.0

        solver = self._solvers_factory.create(params.solver.type, **solver_params)
        solver.train(X=X_train_norm, y=y_train_norm)

        # Create Value Objects
        data_split = DataSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            split_ratio=params.split_ratio,
            random_state=params.random_state,
        )
        transform_pipeline = TransformPipeline(transforms=all_fitted_transforms)

        # Create the engine
        engine = InverseMappingEngine.create(
            dataset_name=params.dataset_name,
            solver=solver,
            transform_pipeline=transform_pipeline,
            data_split=data_split,
        )

        # Assigned version
        version = self._inverse_mapping_engine_repository.save(engine)
        self._logger.log_info(f"InverseMappingEngine saved successfully (v{version}).")

        duration = time.time() - start_time

        transform_summary = [
            f"{t.__class__.__name__}({label})" for label, t in all_fitted_transforms
        ]

        return {
            "dataset_name": params.dataset_name,
            "solver_type": params.solver.type,
            "engine_version": version,
            "status": "completed",
            "duration_seconds": duration,
            "n_train_samples": len(train_indices),
            "n_test_samples": len(test_indices),
            "split_ratio": params.split_ratio,
            "transform_summary": transform_summary,
            "training_history": solver.history(),
        }
