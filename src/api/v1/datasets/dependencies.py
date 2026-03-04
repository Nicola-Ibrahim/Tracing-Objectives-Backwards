from ....modules.dataset.application.delete_dataset import DeleteDatasetService
from ....modules.dataset.application.generation import GenerateDatasetService
from ....modules.dataset.application.get_dataset import GetDatasetDetailsService
from ....modules.dataset.application.list_datasets import ListDatasetsService
from ....modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ....modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ....modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_dataset_repository() -> FileSystemDatasetRepository:
    return FileSystemDatasetRepository()


def get_engine_repository() -> FileSystemInverseMappingEngineRepository:
    return FileSystemInverseMappingEngineRepository()


def get_generation_dataset_service() -> GenerateDatasetService:
    logger = CMDLogger(name="DatasetGenAPI")
    return GenerateDatasetService(
        data_model_repository=get_dataset_repository(),
        logger=logger,
    )


def get_list_datasets_service() -> ListDatasetsService:
    return ListDatasetsService(
        repository=get_dataset_repository(),
        engine_repository=get_engine_repository(),
    )


def get_dataset_details_service() -> GetDatasetDetailsService:
    return GetDatasetDetailsService(
        repository=get_dataset_repository(),
        engine_repository=get_engine_repository(),
    )


def get_delete_dataset_service() -> DeleteDatasetService:
    return DeleteDatasetService(
        repository=get_dataset_repository(),
        engine_repository=get_engine_repository(),
    )
