from fastapi import Depends

from ....modules.modeling.application.transformation_service import (
    TransformationService,
)
from ....modules.modeling.infrastructure.factories.transformer import TransformerFactory
from ..datasets.dependencies import get_dataset_repository


def get_transformation_service(
    repository=Depends(get_dataset_repository),
) -> TransformationService:
    factory = TransformerFactory()
    return TransformationService(transformer_factory=factory, repository=repository)
