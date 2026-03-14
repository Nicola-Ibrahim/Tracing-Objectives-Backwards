from dependency_injector import containers, providers

from ..application.dataset_service import DatasetService
from .repositories.dataset_repository import FileSystemDatasetRepository
from .sources.factory import DataGeneratorFactory


class DatasetContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the Dataset bounded context.
    Using dependency-injector library.
    """

    engine_repository = providers.Dependency()
    logger = providers.Dependency()

    repository = providers.Singleton(FileSystemDatasetRepository)

    generator_factory = providers.Singleton(DataGeneratorFactory)

    dataset_service = providers.Factory(
        DatasetService,
        repository=repository,
        engine_repository=engine_repository,
        generator_factory=generator_factory,
        logger=logger,
    )
