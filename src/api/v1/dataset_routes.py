import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ..dependencies import get_dataset_repository
from ..schemas.dataset import DatasetDetailResponse, DatasetResponse

router = APIRouter()


@router.get("", response_model=list[str])
async def list_datasets(
    repository: FileSystemDatasetRepository = Depends(get_dataset_repository),
):
    """
    List all available datasets in the system.
    """
    return repository.list_all()


@router.get("/{dataset_name}", response_model=DatasetResponse)
async def get_dataset_context(
    dataset_name: str,
    repository: FileSystemDatasetRepository = Depends(get_dataset_repository),
):
    """
    Retrieve the original and normalized coordinates for a specific dataset.
    """
    try:
        dataset = repository.load(dataset_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )

    # Extract bounds
    objs = np.atleast_2d(dataset.objectives)
    bounds = {}
    for i in range(objs.shape[1]):
        bounds[f"obj_{i}"] = (float(np.min(objs[:, i])), float(np.max(objs[:, i])))

    return DatasetResponse(
        name=dataset.name,
        original_objectives=[tuple(row) for row in dataset.objectives.tolist()],
        original_decisions=[row.tolist() for row in dataset.decisions],
        bounds=bounds,
    )


@router.get("/{dataset_name}/data", response_model=DatasetDetailResponse)
async def get_dataset_coordinates(
    dataset_name: str,
    repository: FileSystemDatasetRepository = Depends(get_dataset_repository),
):
    """
    Retrieve the raw (X, y) coordinates and bounds for a specific dataset.
    """
    try:
        dataset = repository.load(dataset_name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_name}' not found."
        )

    # Extract bounds
    objs = np.atleast_2d(dataset.objectives)
    bounds = {}
    for i in range(objs.shape[1]):
        bounds[f"obj_{i}"] = (float(np.min(objs[:, i])), float(np.max(objs[:, i])))

    return DatasetDetailResponse(
        name=dataset.name,
        X=[row.tolist() for row in dataset.decisions],
        y=[row.tolist() for row in dataset.objectives],
        bounds=bounds,
    )
