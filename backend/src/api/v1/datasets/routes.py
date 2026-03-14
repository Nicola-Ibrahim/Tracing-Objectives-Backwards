from fastapi import APIRouter, Depends, HTTPException, status

from ....modules.dataset.application.dataset_service import (
    DatasetConfiguration,
    DatasetService,
)
from ....modules.inverse.application.inverse_service import InverseService
from ..inverse.dependencies import get_inverse_service
from ..inverse.schemas import BulkDeleteEnginesRequest, EngineListItem
from .dependencies import get_dataset_service
from .schemas import (
    BulkDeleteDatasetsRequest,
    DatasetDeleteResponse,
    DatasetDetailResponse,
    DatasetGenerationRequest,
    DatasetGenerationResponse,
    DatasetSummary,
    GeneratorsDiscoveryResponse,
)

router = APIRouter()


@router.get("/{dataset_name}/engines", response_model=list[EngineListItem])
async def list_engines_for_dataset(
    dataset_name: str,
    service: InverseService = Depends(get_inverse_service),
):
    """
    List all trained engines for a specific dataset.
    """
    result = service.list_engines(dataset_name)
    return result.match(
        on_success=lambda value: [
            EngineListItem(
                dataset_name=item.get("dataset_name"),
                solver_type=item["solver_type"],
                version=item["version"],
                created_at=item["created_at"],
            )
            for item in value
        ],
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.delete("/{dataset_name}/engines")
async def delete_engines_for_dataset(
    dataset_name: str,
    request: BulkDeleteEnginesRequest,
    service: InverseService = Depends(get_inverse_service),
):
    """
    Delete specific engines for a dataset.
    """
    # Ensure all engines in the request belong to the specified dataset
    for engine in request.engines:
        engine["dataset_name"] = dataset_name

    result = service.delete_engines(request.engines)

    return result.match(
        on_success=lambda value: value["results"],
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.get("/generators", response_model=GeneratorsDiscoveryResponse)
async def list_available_generators(
    service: DatasetService = Depends(get_dataset_service),
):
    """
    List all available dataset generators and their required parameters.
    """
    result = service.get_available_generators()
    return result.match(
        on_success=lambda schemas: GeneratorsDiscoveryResponse(generators=schemas),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    service: DatasetService = Depends(get_dataset_service),
):
    """
    List all available datasets in the system with metadata.
    """
    result = service.list_datasets()
    return result.match(
        on_success=lambda datasets: [
            DatasetSummary(
                name=dataset["name"],
                n_features=dataset["n_features"],
                n_objectives=dataset["n_objectives"],
                metadata=dataset["metadata"],
                trained_engines_count=dataset["trained_engines_count"],
            )
            for dataset in datasets
        ],
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if error.code == "NOT_FOUND"
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.get("/{dataset_name}", response_model=DatasetDetailResponse)
async def get_dataset_details(
    dataset_name: str,
    split: str = "train",
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Retrieve full details for a specific dataset including X, y, Pareto mask, and engines.
    Can be filtered by split (train, test, all).
    """
    result = service.get_dataset_details(dataset_name, split=split)
    return result.match(
        on_success=lambda value: DatasetDetailResponse(
            name=value["name"],
            objectives_dim=value["objectives_dim"],
            decisions_dim=value["decisions_dim"],
            metadata=value["metadata"],
            X=value["X"],
            y=value["y"],
            is_pareto=value["is_pareto"],
            bounds=value["bounds"],
            trained_engines=value["trained_engines"],
        ),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if error.code == "NOT_FOUND"
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.post("", response_model=DatasetGenerationResponse, status_code=201)
async def generate_dataset(
    request: DatasetGenerationRequest,
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Consolidated endpoint to generate a new dataset.
    """
    config = DatasetConfiguration(
        dataset_name=request.dataset_name,
        generator_type=request.generator_type,
        params=request.params,
        split_ratio=request.split_ratio,
        random_state=request.random_state,
    )

    result = service.generate_dataset(config)

    return result.match(
        on_success=lambda path: DatasetGenerationResponse(
            status="success",
            path=str(path),
            name=request.dataset_name,
        ),
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )


@router.delete("", response_model=list[DatasetDeleteResponse])
async def delete_datasets(
    request: BulkDeleteDatasetsRequest,
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Consolidated endpoint to delete one or multiple datasets and all associated trained engines.
    """
    result = service.delete_datasets(request.dataset_names)
    return result.match(
        on_success=lambda items: [
            DatasetDeleteResponse(
                name=item["name"],
                status=item["status"],
                engines_removed=item.get("engines_removed", 0),
            )
            for item in items
        ],
        on_failure=lambda error: HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        ),
    )
