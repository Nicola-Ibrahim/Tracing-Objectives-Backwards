from ...inverse.domain.interfaces.base_inverse_mapping_engine_repository import (
    BaseInverseMappingEngineRepository,
)
from ..domain.interfaces.base_repository import BaseDatasetRepository


class DeleteDatasetService:
    """
    Application service to delete a dataset and its associated trained engines.
    """

    def __init__(
        self,
        repository: BaseDatasetRepository,
        engine_repository: BaseInverseMappingEngineRepository,
    ):
        self._repository = repository
        self._engine_repository = engine_repository

    def execute(self, dataset_name: str) -> dict:
        # Check if dataset exists
        self._repository.load(dataset_name)

        # Delete dataset
        self._repository.delete(dataset_name)

        # Delete engines
        engines_removed = self._engine_repository.delete_all_for_dataset(dataset_name)

        return {"name": dataset_name, "engines_removed": engines_removed}
