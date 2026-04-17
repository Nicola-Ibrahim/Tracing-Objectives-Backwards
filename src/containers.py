from dependency_injector import containers, providers

from .modules.dataset.infrastructure.config.di import DatasetContainer
from .modules.evaluation.infrastructure.config.di import EvaluationContainer
from .modules.inverse.infrastructure.config.di import InverseContainer
from .modules.modeling.infrastructure.config.di import ModelingContainer
from .modules.shared.infrastructure.loggers.cmd_logger import CMDLogger
from .modules.shared.infrastructure.redis.connection import RedisConnection
from .modules.shared.infrastructure.redis.task_manager import RedisTaskManager


class RootContainer(containers.DeclarativeContainer):
    """
    Root Dependency Injection container.
    Composes all module sub-containers and global resources.
    """

    # Configuration providers
    config = providers.Configuration()

    # Global singletons
    logger = providers.Singleton(CMDLogger, name="backend")
    redis_connection = providers.Singleton(RedisConnection, url=config.redis_url)

    # Unified task management infrastructure
    task_manager = providers.Singleton(RedisTaskManager, connection=redis_connection)

    # Module sub-containers (composed)
    # We pass global dependencies directly into sub-containers
    inverse = providers.Container(
        InverseContainer,
        logger=logger,
    )
    dataset = providers.Container(
        DatasetContainer,
        logger=logger,
        engine_repository=inverse.repository,
    )

    # TODO: we need to move the dependency on inverse repo in dataset and
    # create two separate requests at the API level.
    inverse.dataset_repository.override(dataset.repository)

    evaluation = providers.Container(
        EvaluationContainer,
        logger=logger,
        engine_repository=inverse.repository,
        data_repository=dataset.repository,
        # Unified port replaces fragmented ones
        task_manager=task_manager,
    )

    modeling = providers.Container(
        ModelingContainer,
        logger=logger,
        dataset_repository=dataset.repository,
    )
