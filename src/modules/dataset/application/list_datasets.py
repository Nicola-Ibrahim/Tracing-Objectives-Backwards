import numpy as np

from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.interfaces.base_repository import BaseDatasetRepository


class ListDatasetsService:
    """
    Application service to list all datasets with their summary statistics.
    """

    def __init__(
        self,
        repository: BaseDatasetRepository,
        engine_repository: BaseInverseMappingEngineRepository,
    ):
        self._repository = repository
        self._engine_repository = engine_repository

    def execute(self) -> list[dict]:
        names = self._repository.list_all()
        summaries = []
        for name in names:
            try:
                dataset = self._repository.load(name)
                # Check training status
                engines_meta = self._engine_repository.list_engines(name)

                # Extract stats
                objs = np.atleast_2d(dataset.objectives)
                decs = np.atleast_2d(dataset.decisions)

                summaries.append(
                    {
                        "name": name,
                        "n_samples": objs.shape[0],
                        "n_features": decs.shape[1],
                        "n_objectives": objs.shape[1],
                        "trained_engines_count": len(engines_meta),
                    }
                )
            except Exception:
                # Skip corrupted or partially deleted datasets
                continue
        return summaries
