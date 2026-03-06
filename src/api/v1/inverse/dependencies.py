from ....modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ....modules.inverse.application.inverse_service import InverseService
from ....modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ....modules.inverse.infrastructure.solvers.factory import SolversFactory
from ....modules.modeling.infrastructure.factories.transformer import TransformerFactory
from ....modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_solvers_factory() -> SolversFactory:
    return SolversFactory()


def get_inverse_mapping_engine_repository() -> FileSystemInverseMappingEngineRepository:
    return FileSystemInverseMappingEngineRepository()


def get_inverse_service() -> InverseService:
    logger = CMDLogger(name="InverseAPI")
    return InverseService(
        dataset_repository=FileSystemDatasetRepository(),
        inverse_mapping_engine_repository=get_inverse_mapping_engine_repository(),
        logger=logger,
        transformer_factory=TransformerFactory(),
        solvers_factory=get_solvers_factory(),
    )
