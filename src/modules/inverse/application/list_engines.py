from ..domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)


class ListEnginesService:
    """
    Application service to list all trained engines for a dataset.
    """

    def __init__(
        self,
        repository: BaseInverseMappingEngineRepository,
    ):
        self._repository = repository

    def execute(self, dataset_name: str) -> list[dict]:
        engines = self._repository.list_all(dataset_name)
        return [
            {
                "solver_type": e["solver_type"],
                "version": e["version"],
                "created_at": e["created_at"],
            }
            for e in engines
        ]
