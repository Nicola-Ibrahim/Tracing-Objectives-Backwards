from ..modules.dataset.application.generation import GenerateDatasetService
from ..modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ..modules.generation.application.generate_candidates import (
    GenerateCoherentCandidatesService,
)
from ..modules.generation.infrastructure.repositories.context_repo import (
    FileSystemContextRepository,
)
from ..modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_dataset_repository() -> FileSystemDatasetRepository:
    return FileSystemDatasetRepository()


def get_context_repository() -> FileSystemContextRepository:
    return FileSystemContextRepository()


def get_generation_service() -> GenerateCoherentCandidatesService:
    logger = CMDLogger(name="GenerationAPI")
    return GenerateCoherentCandidatesService(
        context_repository=get_context_repository(),
        dataset_repository=get_dataset_repository(),
        logger=logger,
    )


def get_generation_dataset_service() -> GenerateDatasetService:
    logger = CMDLogger(name="DatasetGenAPI")
    return GenerateDatasetService(
        data_model_repository=get_dataset_repository(),
        logger=logger,
    )
