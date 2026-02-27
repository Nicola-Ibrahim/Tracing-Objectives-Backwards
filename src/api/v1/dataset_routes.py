import numpy as np
from fastapi import APIRouter, HTTPException

from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ..schemas.dataset import DatasetResponse

router = APIRouter()
repository = FileSystemDatasetRepository()


@router.get("/", response_model=list[str])
async def list_datasets():
    """
    List all available datasets in the system.
    """
    return repository.list_all()


@router.get("/{dataset_name}", response_model=DatasetResponse)
async def get_dataset_context(dataset_name: str):
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
