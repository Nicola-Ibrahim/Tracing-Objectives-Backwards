import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...modules.dataset.application.generation import (
    DatasetConfiguration,
    GenerateDatasetService,
)
from ...modules.dataset.infrastructure.repositories.dataset_repository import (
    FileSystemDatasetRepository,
)
from ...modules.generation.domain.interfaces.base_context_repository import (
    BaseContextRepository,
)
from ..dependencies import (
    get_context_repository,
    get_dataset_repository,
    get_generation_dataset_service,
)
from ..schemas.dataset import (
    DatasetDetailResponse,
    DatasetGenerationRequest,
    DatasetResponse,
)

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
    context_repository: BaseContextRepository = Depends(get_context_repository),
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

    # Check training status
    is_trained = False
    try:
        context = context_repository.load(dataset_name)
        is_trained = getattr(context, "is_trained", True)
    except FileNotFoundError:
        is_trained = False

    return DatasetResponse(
        name=dataset.name,
        original_objectives=[tuple(row) for row in dataset.objectives.tolist()],
        original_decisions=[row.tolist() for row in dataset.decisions],
        bounds=bounds,
        is_trained=is_trained,
    )


@router.get("/{dataset_name}/data", response_model=DatasetDetailResponse)
async def get_dataset_coordinates(
    dataset_name: str,
    repository: FileSystemDatasetRepository = Depends(get_dataset_repository),
    context_repository: BaseContextRepository = Depends(get_context_repository),
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

    # Calculate Pareto mask by comparing with the stored pareto front if available
    is_pareto = [False] * objs.shape[0]
    if dataset.pareto is not None and dataset.pareto.front is not None:
        # Simple exact match for pre-computed pareto front
        pareto_front = dataset.pareto.front
        for i, obj in enumerate(objs):
            # Using isclose to handle potential floating point precision issues
            if any(np.allclose(obj, p_obj) for p_obj in pareto_front):
                is_pareto[i] = True

    # Check training status
    is_trained = False
    try:
        context = context_repository.load(dataset_name)
        is_trained = getattr(context, "is_trained", True)
    except FileNotFoundError:
        is_trained = False

    return DatasetDetailResponse(
        name=dataset.name,
        X=[row.tolist() for row in dataset.decisions],
        y=[row.tolist() for row in dataset.objectives],
        is_pareto=is_pareto,
        bounds=bounds,
        is_trained=is_trained,
    )


@router.post("/generate")
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
    )

    try:
        path = service.execute(config)
        return {"status": "success", "path": str(path), "name": request.dataset_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
