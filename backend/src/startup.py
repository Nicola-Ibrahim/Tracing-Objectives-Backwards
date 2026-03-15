from dependency_injector import containers, providers

from .modules.dataset.infrastructure.di import DatasetContainer
from .modules.evaluation.infrastructure.di import EvaluationContainer
from .modules.inverse.infrastructure.di import InverseContainer
from .modules.modeling.infrastructure.di import ModelingContainer
from .modules.shared.infrastructure.loggers.cmd_logger import CMDLogger


class ModulesContainer(containers.DeclarativeContainer):
    """
    Global Dependency Injection container that aggregates all module-level containers.
    Using dependency-injector library.
    """

    # Shared resources
    logger = providers.Singleton(CMDLogger, name="TracingObjectivesBackwards")

    # Module Containers
    modeling = providers.Container(
        ModelingContainer,
        # modeling needs dataset_repository (interface only, wired later)
    )

    inverse = providers.Container(
        InverseContainer,
        logger=logger,
        # inverse needs dataset_repository (interface only, wired later)
    )

    dataset = providers.Container(
        DatasetContainer,
        logger=logger,
        # dataset needs engine_repository (interface only, wired later)
    )

    evaluation = providers.Container(
        EvaluationContainer,
        logger=logger,
        # evaluation needs engine_repository and data_repository
        # (interfaces only, wired later)
    )


async def start_backend(container: ModulesContainer):
    """Starts any resources that require async initialization."""
    # Note: dependency-injector doesn't have a built-in async start,
    # but we can implement it here.
    # container.logger().log_info("Backend Services Starting...")
    pass


async def stop_backend(container: ModulesContainer):
    """Gracefully shuts down resources."""
    # container.logger().log_info("Backend Services Shutting Down...")
    pass


def initialize_modules():
    """
    Wires cross-module dependencies using dependency-injector wiring features.
    """
    container = ModulesContainer()

    # Wire Modeling dependencies
    container.modeling.dataset_repository.override(container.dataset.repository)

    # Wire Inverse dependencies
    container.inverse.dataset_repository.override(container.dataset.repository)

    # Wire Dataset dependencies
    container.dataset.engine_repository.override(container.inverse.repository)

    # Wire Evaluation dependencies
    container.evaluation.engine_repository.override(container.inverse.repository)
    container.evaluation.data_repository.override(container.dataset.repository)

    return container


# Global container instance
ModulesContainer = initialize_modules()
