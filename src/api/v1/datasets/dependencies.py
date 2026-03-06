from ....modules.dataset.application.dataset_service import DatasetService
from ....modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ....modules.dataset.infrastructure.sources.factory import DataGeneratorFactory
from ....modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ....modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_generator_factory() -> DataGeneratorFactory:
    return DataGeneratorFactory()


def get_dataset_repository() -> FileSystemDatasetRepository:
    return FileSystemDatasetRepository()


def get_engine_repository() -> FileSystemInverseMappingEngineRepository:
    return FileSystemInverseMappingEngineRepository()


def get_dataset_service() -> DatasetService:
    logger = CMDLogger(name="DatasetAPI")
    return DatasetService(
        repository=get_dataset_repository(),
        engine_repository=get_engine_repository(),
        generator_factory=DataGeneratorFactory(),
        logger=logger,
    )
