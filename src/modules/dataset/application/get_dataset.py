import numpy as np

from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.interfaces.base_repository import BaseDatasetRepository


class GetDatasetDetailsService:
    """
    Application service to retrieve full dataset details including X, y, Pareto mask, and engines.
    """

    def __init__(
        self,
        repository: BaseDatasetRepository,
        engine_repository: BaseInverseMappingEngineRepository,
    ):
        self._repository = repository
        self._engine_repository = engine_repository

    def execute(self, dataset_name: str, split: str = "train") -> dict:
        dataset = self._repository.load(dataset_name)

        # Handle split slicing
        if split == "train":
            decisions, objectives = dataset.get_train_data()
        elif split == "test":
            decisions, objectives = dataset.get_test_data()
        else:
            # "all" or invalid (defaults to all if not explicitly train/test in this logic)
            decisions, objectives = dataset.decisions, dataset.objectives

        # Extract bounds (from the REQUESTED split to match visual domain)
        objs = np.atleast_2d(objectives)
        bounds = {}
        if objs.size > 0:
            for i in range(objs.shape[1]):
                bounds[f"obj_{i}"] = (
                    float(np.min(objs[:, i])),
                    float(np.max(objs[:, i])),
                )

        # Calculate Pareto mask for the subset
        is_pareto = [False] * objectives.shape[0]
        if dataset.pareto is not None and dataset.pareto.front is not None:
            pareto_front = dataset.pareto.front
            for i, obj in enumerate(objectives):
                if any(np.allclose(obj, p_obj) for p_obj in pareto_front):
                    is_pareto[i] = True

        # Check training status
        engines_meta = self._engine_repository.list_engines(dataset_name)
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
