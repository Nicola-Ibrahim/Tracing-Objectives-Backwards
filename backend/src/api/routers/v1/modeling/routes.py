from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, status

from src.containers import RootContainer
from src.modules.modeling.application.transformation_service import (
    TransformationService,
)

from .schemas import (
    DataPreviewPoints,
    TransformationPreviewRequest,
    TransformationPreviewResponse,
    TransformerRegistryResponse,
)

router = APIRouter(prefix="/modeling", tags=["Modeling"])


@router.get("/transformers", response_model=TransformerRegistryResponse)
@inject
async def list_transformers(
    service: Annotated[
        TransformationService,
        Depends(Provide[RootContainer.modeling.transformation_service]),
    ],
):
    """
    Returns a registry of all available data transformers (normalizers, encoders, etc).
    Each transformer includes a schema of required/optional parameters.
    """
    result = service.get_available_transformers()
    return result.match(
        on_success=lambda value: TransformerRegistryResponse(transformers=value),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.post("/transform", response_model=TransformationPreviewResponse)
@inject
async def preview_transformation(
    request: TransformationPreviewRequest,
    service: Annotated[
        TransformationService,
        Depends(Provide[RootContainer.modeling.transformation_service]),
    ],
):
    """
    Applies a series of transformations to a small dataset sample and returns
    before/after statistics and sample points.
    Used for UI previewing of transformation chains.
    """
    # Use request.x_chain and request.y_chain instead of legacy request.chains
    result = service.get_transformation_preview(
        dataset_name=request.dataset_name,
        x_chain=[s.dict() for s in request.x_chain],
        y_chain=[s.dict() for s in request.y_chain],
        split=request.split,
        sampling_limit=request.sampling_limit,
    )

    return result.match(
        on_success=lambda value: TransformationPreviewResponse(
            original=DataPreviewPoints(X=value["X_original"], y=value["y_original"]),
            transformed=DataPreviewPoints(
                X=value["X_transformed"], y=value["y_transformed"]
            ),
            metrics=value["metrics"],
        ),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            if error.code == "VALIDATION_ERROR"
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )
