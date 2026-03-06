from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ...shared.domain.interfaces.base_logger import BaseLogger
from ..domain.entities.dataset import Dataset
from ..domain.interfaces.base_repository import BaseDatasetRepository
from ..domain.interfaces.base_visualizer import BaseVisualizer
from ..domain.value_objects.pareto import Pareto
from ..infrastructure.sources.factory import DataGeneratorFactory


class DatasetConfiguration(BaseModel):
    """Unified configuration for dataset generation."""

    generator_type: str = Field("coco_pymoo", description="Type of generator to use.")
    dataset_name: str = Field(..., description="Identifier for the dataset.")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Generator-specific parameters."
    )
    split_ratio: float = Field(
        0.2, ge=0.0, lt=1.0, description="Ratio of data used for testing."
    )
    random_state: int = Field(42, description="Random seed for reproducibility.")


class DatasetService:
    """
    Consolidated application service for dataset management, generation, and visualization.
    """

    def __init__(
        self,
        repository: BaseDatasetRepository,
        engine_repository: BaseInverseMappingEngineRepository,
        generator_factory: DataGeneratorFactory,
        logger: BaseLogger,
        visualizer: BaseVisualizer | None = None,
    ):
        self._repository = repository
        self._engine_repository = engine_repository
        self._generator_factory = generator_factory
        self._logger = logger
        self._visualizer = visualizer

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Lists all datasets with their summary statistics."""
        names = self._repository.list_all()
        summaries = []
        for name in names:
            try:
                dataset = self._repository.load(name)
                engines_meta = self._engine_repository.list_engines(name)

                objs = np.atleast_2d(dataset.y)
                decs = np.atleast_2d(dataset.X)

                summaries.append(
                    {
                        "name": name,
                        "n_samples": objs.shape[0],
                        "n_features": decs.shape[1],
                        "n_objectives": objs.shape[1],
                        "trained_engines_count": len(engines_meta),
                    }
                )
            except Exception as e:
                self._logger.log_error(f"Failed to load dataset {name}: {e}")
                continue
        return summaries

    def get_dataset_details(self, name: str, split: str = "train") -> Dict[str, Any]:
        """Retrieves full dataset details including X, y, Pareto mask, and engines."""
        dataset = self._repository.load(name)

        if split == "train":
            decisions, objectives = dataset.get_train_data()
        elif split == "test":
            decisions, objectives = dataset.get_test_data()
        else:
            decisions, objectives = dataset.X, dataset.y

        objs = np.atleast_2d(objectives)
        bounds = {}
        if objs.size > 0:
            for i in range(objs.shape[1]):
                bounds[f"obj_{i}"] = (
                    float(np.min(objs[:, i])),
                    float(np.max(objs[:, i])),
                )

        is_pareto = [False] * objectives.shape[0]
        if dataset.pareto is not None and dataset.pareto.front is not None:
            pareto_front = dataset.pareto.front
            for i, obj in enumerate(objectives):
                if any(np.allclose(obj, p_obj) for p_obj in pareto_front):
                    is_pareto[i] = True

        engines_meta = self._engine_repository.list_engines(name)
        trained_engines = [
            {
                "solver_type": e["solver_type"],
                "version": e["version"],
                "created_at": e["created_at"],
            }
            for e in engines_meta
        ]

        return {
            "name": dataset.name,
            "samples": len(objectives),
            "objectives_count": objectives.shape[1] if objectives.size > 0 else 0,
            "decisions_count": decisions.shape[1] if decisions.size > 0 else 0,
            "X": [row.tolist() for row in decisions],
            "y": [row.tolist() for row in objectives],
            "is_pareto": is_pareto,
            "bounds": bounds,
            "trained_engines": trained_engines,
        }

    def delete_datasets(self, names: List[str]) -> List[Dict[str, Any]]:
        """Deletes multiple datasets and their associated trained engines."""
        results = []
        for name in names:
            try:
                self._repository.load(name)
                self._repository.delete(name)
                engines_removed = self._engine_repository.delete_all_for_dataset(name)
                results.append(
                    {
                        "name": name,
                        "engines_removed": engines_removed,
                        "status": "deleted",
                    }
                )
            except FileNotFoundError:
                results.append({"name": name, "status": "not_found"})
            except Exception as e:
                results.append({"name": name, "status": "error", "error": str(e)})
        return results

    def generate_dataset(self, config: DatasetConfiguration) -> Path:
        """Orchestrates dataset generation using the factory."""
        self._logger.log_info(
            f"Starting dataset generation with {config.generator_type} for dataset {config.dataset_name}"
        )

        data_source = self._generator_factory.create(
            generator_type=config.generator_type, params=config.params
        )

        raw_data = data_source.generate()

        pareto = None
        if raw_data.pareto_set is not None and raw_data.pareto_front is not None:
            pareto = Pareto.create(
                set=raw_data.pareto_set,
                front=raw_data.pareto_front,
            )

        dataset = Dataset.create(
            name=config.dataset_name,
            X=raw_data.decisions,
            y=raw_data.objectives,
            pareto=pareto,
            split_ratio=config.split_ratio,
            random_state=config.random_state,
        )

        saved_path = self._repository.save(dataset)
        self._logger.log_info(f"Dataset saved to: {saved_path}")
        return saved_path

    def visualize_dataset(self, name: str) -> None:
        """Visualizes a dataset if a visualizer is provided."""
        if not self._visualizer:
            raise RuntimeError("Visualizer not initialized in DatasetService")

        dataset = self._repository.load(name=name)
        payload = {
            "dataset_name": dataset.name,
            "X_raw": dataset.X,
            "y_raw": dataset.y,
            "pareto_set": dataset.pareto.set if dataset.pareto else None,
            "pareto_front": dataset.pareto.front if dataset.pareto else None,
        }
        self._visualizer.plot(data=payload)
