from typing import Any, Dict, List

import numpy as np

from ...dataset.domain.interfaces.base_repository import BaseDatasetRepository
from ...shared.result import Result
from ..domain.services.transformation_domain_service import (
    ITransformerFactory,
    TransformationConfig,
    TransformationDomainService,
)


class TransformationService:
    """
    Application service that manages transformation chains for datasets.
    """

    def __init__(
        self,
        transformer_factory: ITransformerFactory,
        transformation_domain_service: TransformationDomainService,
        repository: BaseDatasetRepository,
    ):
        self._transformer_factory = transformer_factory
        self._transformation_domain_service = transformation_domain_service
        self._repository = repository

    def get_available_transformers(self) -> Result[List[Dict[str, Any]]]:
        """Returns metadata for all supported transformers."""
        try:
            return Result.ok(self._transformer_factory.get_transformer_schemas())
        except Exception as e:
            return Result.fail(
                message="Failed to get available transformers",
                details=str(e),
                code="INTERNAL_ERROR",
            )

    def calculate_preview_metrics(
        self,
        X_orig: np.ndarray,
        y_orig: np.ndarray,
        X_trans: np.ndarray,
        y_trans: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates basic summary stats for before/after comparison."""
        metrics = {}

        if X_orig.size > 0:
            metrics["X_min_orig"] = X_orig.min(axis=0).tolist()
            metrics["X_max_orig"] = X_orig.max(axis=0).tolist()

        if X_trans.size > 0:
            metrics["X_min_trans"] = X_trans.min(axis=0).tolist()
            metrics["X_max_trans"] = X_trans.max(axis=0).tolist()

        if y_orig.size > 0:
            metrics["y_min_orig"] = y_orig.min(axis=0).tolist()
            metrics["y_max_orig"] = y_orig.max(axis=0).tolist()

        if y_trans.size > 0:
            metrics["y_min_trans"] = y_trans.min(axis=0).tolist()
            metrics["y_max_trans"] = y_trans.max(axis=0).tolist()

        return metrics

    def get_transformation_preview(
        self,
        dataset_name: str,
        x_chain: List[Dict[str, Any]],
        y_chain: List[Dict[str, Any]],
        split: str = "train",
        sampling_limit: int = 2000,
    ) -> Result[Dict[str, Any]]:
        """
        Orchestrates data loading, sampling, and transformation application for preview.
        """
        try:
            # 1. Load data
            dataset = self._repository.load(dataset_name)

            if split == "train":
                X_raw, y_raw = dataset.get_train_data()
            elif split == "test":
                X_raw, y_raw = dataset.get_test_data()
            else:
                X_raw, y_raw = dataset.X, dataset.y

            X = np.array(X_raw)
            y = np.array(y_raw)

            if X.shape[0] == 0:
                return Result.fail("Dataset is empty.", code="VALIDATION_ERROR")

            # 2. Sample data for performance
            if X.shape[0] > sampling_limit:
                indices = np.random.choice(X.shape[0], sampling_limit, replace=False)
                X = X[indices]
                y = y[indices]

            # 3. Apply transformations using Domain Service
            x_configs = [TransformationConfig(**c) for c in x_chain]
            y_configs = [TransformationConfig(**c) for c in y_chain]

            transformed = self._transformation_domain_service.apply_chains(
                X, y, x_configs, y_configs
            )
            X_trans, y_trans = transformed.X, transformed.y

            # 4. Calculate metrics
            metrics = self.calculate_preview_metrics(X, y, X_trans, y_trans)

            return Result.ok(
                {
                    "X_original": X.tolist(),
                    "y_original": y.tolist(),
                    "X_transformed": X_trans.tolist(),
                    "y_transformed": y_trans.tolist(),
                    "metrics": metrics,
                }
            )
        except ValueError as e:
            return Result.fail(
                message="Failed to get transformation preview",
                details=str(e),
                code="VALIDATION_ERROR",
            )
        except Exception as e:
            return Result.fail(
                message="Failed to get transformation preview",
                details=str(e),
                code="INTERNAL_ERROR",
            )
