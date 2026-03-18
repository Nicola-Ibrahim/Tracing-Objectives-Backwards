from dependency_injector import containers, providers

from ....modeling.infrastructure.factories.transformer import TransformerFactory
from ...application.inverse_service import InverseService
from ..repositories.inverse_mapping_engine_repo import (
    FileSystemInverseMappingEngineRepository,
)
from ..solvers.factory import SolversFactory


class InverseContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the Inverse bounded context.
    Using dependency-injector library.
    """

    logger = providers.Dependency()
    dataset_repository = providers.Dependency()

    transformer_factory = providers.Singleton(TransformerFactory)

    repository = providers.Singleton(
        FileSystemInverseMappingEngineRepository,
        transformer_factory=transformer_factory,
    )

    solvers_factory = providers.Singleton(SolversFactory)

    inverse_service = providers.Factory(
        InverseService,
        dataset_repository=dataset_repository,
        inverse_mapping_engine_repository=repository,
        logger=logger,
        transformer_factory=transformer_factory,
        solvers_factory=solvers_factory,
    )
