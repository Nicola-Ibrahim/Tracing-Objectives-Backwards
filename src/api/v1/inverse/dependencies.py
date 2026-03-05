from ....modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ....modules.inverse.application.generate_candidates import (
    GenerateCandidatesService,
)
from ....modules.inverse.application.list_engines import ListEnginesService
from ....modules.inverse.application.train_inverse_mapping_engine import (
    TrainInverseMappingEngineService,
)
from ....modules.inverse.infrastructure.repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ....modules.inverse.infrastructure.solvers.factory import SolversFactory
from ....modules.modeling.infrastructure.factories.transformer import TransformerFactory
from ....modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


def get_inverse_mapping_engine_repository() -> FileSystemInverseMappingEngineRepository:
    return FileSystemInverseMappingEngineRepository()


def get_train_service() -> TrainInverseMappingEngineService:
    logger = CMDLogger(name="InverseTrainingAPI")
    return TrainInverseMappingEngineService(
        dataset_repository=FileSystemDatasetRepository(),
        inverse_mapping_engine_repository=get_inverse_mapping_engine_repository(),
        logger=logger,
        transformer_factory=TransformerFactory(),
        solvers_factory=SolversFactory(),
    )


def get_generation_service() -> GenerateCandidatesService:
    logger = CMDLogger(name="InverseGenerationAPI")
    return GenerateCandidatesService(
        inverse_mapping_engine_repository=get_inverse_mapping_engine_repository(),
        logger=logger,
    )


def get_list_engines_service() -> ListEnginesService:
    return ListEnginesService(repository=get_inverse_mapping_engine_repository())
