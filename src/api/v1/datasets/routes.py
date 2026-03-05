from fastapi import APIRouter, Depends, HTTPException

from ....modules.dataset.application.delete_dataset import DeleteDatasetService
from ....modules.dataset.application.generation import (
    DatasetConfiguration,
    GenerateDatasetService,
)
from ....modules.dataset.application.get_dataset import GetDatasetDetailsService
from ....modules.dataset.application.list_datasets import ListDatasetsService
from .dependencies import (
    get_dataset_details_service,
    get_delete_dataset_service,
    get_generation_dataset_service,
    get_list_datasets_service,
)
from .schemas import (
    DatasetDeleteResponse,
    DatasetDetailResponse,
    DatasetGenerationRequest,
    DatasetGenerationResponse,
    DatasetSummary,
)

router = APIRouter()


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    service: ListDatasetsService = Depends(get_list_datasets_service),
):
    """
    List all available datasets in the system with metadata.
    """
    return service.execute()


@router.get("/{dataset_name}", response_model=DatasetDetailResponse)
async def get_dataset_details(
    dataset_name: str,
    split: str = "train",
    service: GetDatasetDetailsService = Depends(get_dataset_details_service),
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
        return service.execute(dataset_name, split=split)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )


@router.post("/generate", response_model=DatasetGenerationResponse, status_code=201)
async def generate_dataset(
    request: DatasetGenerationRequest,
    service: GenerateDatasetService = Depends(get_generation_dataset_service),
):
    """
    Generate a new COCOEX dataset.
    """
    config = DatasetConfiguration(
        problem_id=request.function_id,
        n_var=request.n_var,
        population_size=request.population_size,
        generations=request.generations,
        dataset_name=request.dataset_name,
        split_ratio=request.split_ratio,
        random_state=request.random_state,
    )

    try:
        path = service.execute(config)
        return {"status": "success", "path": str(path), "name": request.dataset_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_name}", response_model=DatasetDeleteResponse)
async def delete_dataset(
    dataset_name: str,
    service: DeleteDatasetService = Depends(get_delete_dataset_service),
):
    """
    Delete a dataset and all associated trained engines.
    """
    try:
        return service.execute(dataset_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )
