from dependency_injector import containers, providers

from ...application.transformation_service import TransformationService
from ...domain.services.transformation_domain_service import TransformationDomainService
from ..factories.transformer import TransformerFactory


class ModelingContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the Modeling bounded context.
    Using dependency-injector library.
    """

    logger = providers.Dependency()
    dataset_repository = providers.Dependency()

    transformer_factory = providers.Singleton(TransformerFactory)

    transformation_domain_service = providers.Singleton(
        TransformationDomainService,
        transformer_factory=transformer_factory,
    )

    transformation_service = providers.Factory(
        TransformationService,
        transformer_factory=transformer_factory,
        transformation_domain_service=transformation_domain_service,
        repository=dataset_repository,
    )
