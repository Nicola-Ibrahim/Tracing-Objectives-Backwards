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

    def execute(self, dataset_name: str) -> dict:
        dataset = self._repository.load(dataset_name)

        # Extract bounds
        objs = np.atleast_2d(dataset.objectives)
        bounds = {}
        for i in range(objs.shape[1]):
            bounds[f"obj_{i}"] = (float(np.min(objs[:, i])), float(np.max(objs[:, i])))

        # Calculate Pareto mask by comparing with the stored pareto front if available
        is_pareto = [False] * objs.shape[0]
        if dataset.pareto is not None and dataset.pareto.front is not None:
            # Simple exact match for pre-computed pareto front
            pareto_front = dataset.pareto.front
            for i, obj in enumerate(objs):
                # Using isclose to handle potential floating point precision issues
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
            "X": [row.tolist() for row in dataset.decisions],
            "y": [row.tolist() for row in dataset.objectives],
            "is_pareto": is_pareto,
            "bounds": bounds,
            "trained_engines": trained_engines,
        }
