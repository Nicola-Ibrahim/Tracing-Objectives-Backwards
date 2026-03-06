from fastapi import APIRouter, Depends, HTTPException

from ....modules.dataset.application.dataset_service import (
    DatasetConfiguration,
    DatasetService,
)
from ....modules.dataset.infrastructure.sources.factory import DataGeneratorFactory
from .dependencies import get_dataset_service, get_generator_factory
from .schemas import (
    DatasetDeleteResponse,
    DatasetDetailResponse,
    DatasetGenerationRequest,
    DatasetGenerationResponse,
    DatasetSummary,
    GeneratorsDiscoveryResponse,
)

router = APIRouter()


@router.get("/generators", response_model=GeneratorsDiscoveryResponse)
async def list_available_generators(
    factory: DataGeneratorFactory = Depends(get_generator_factory),
):
    """
    List all available dataset generators and their required parameters.
    """
    schemas = factory.get_generator_schemas()
    return GeneratorsDiscoveryResponse(generators=schemas)


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    service: DatasetService = Depends(get_dataset_service),
):
    """
    List all available datasets in the system with metadata.
    """
    return service.list_datasets()


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
    try:
        if split not in ["train", "test", "all"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid split parameter. Must be 'train', 'test', or 'all'.",
            )
        return service.get_dataset_details(dataset_name, split=split)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )


@router.post("/generate", response_model=DatasetGenerationResponse, status_code=201)
async def generate_dataset(
    request: DatasetGenerationRequest,
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Generate a new COCOEX dataset.
    """
    config = DatasetConfiguration(
        dataset_name=request.dataset_name,
        generator_type=request.generator_type,
        params=request.params,
        split_ratio=request.split_ratio,
        random_state=request.random_state,
    )

    try:
        path = service.generate_dataset(config)
        return {"status": "success", "path": str(path), "name": request.dataset_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_name}", response_model=DatasetDeleteResponse)
async def delete_dataset(
    dataset_name: str,
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Delete a dataset and all associated trained engines.
    """
    try:
        return service.delete_dataset(dataset_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )
