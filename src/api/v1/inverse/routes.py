from fastapi import APIRouter, Depends, HTTPException

from ....modules.inverse.application.inverse_service import (
    GenerationConfig,
    InverseService,
    SolverConfig,
    TrainInverseMappingEngineParams,
)
from ....modules.inverse.infrastructure.solvers.factory import SolversFactory
from .dependencies import get_inverse_service, get_solvers_factory
from .schemas import (
    BulkDeleteEnginesRequest,
    EngineListItem,
    GenerateRequest,
    GenerateResponse,
    SolversDiscoveryResponse,
    TrainEngineRequest,
    TrainEngineResponse,
)

router = APIRouter()


@router.get("/solvers", response_model=SolversDiscoveryResponse)
async def list_available_solvers(
    factory: SolversFactory = Depends(get_solvers_factory),
):
    """
    List all available inverse mapping solvers and their required parameters.
    """
    schemas = factory.get_solver_schemas()
    return SolversDiscoveryResponse(solvers=schemas)


@router.post("/train", response_model=TrainEngineResponse, status_code=201)
async def train_engine(
    request: TrainEngineRequest,
    service: InverseService = Depends(get_inverse_service),
):
    """
    Train an inverse mapping engine for a dataset.
    """
    params = TrainInverseMappingEngineParams(
        dataset_name=request.dataset_name,
        solver=SolverConfig(type=request.solver.type, params=request.solver.params),
        transforms=request.transforms,
    )

    try:
        result = service.train_engine(params)
        return TrainEngineResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/generate", response_model=GenerateResponse)
async def generate_candidates(
    request: GenerateRequest,
    service: InverseService = Depends(get_inverse_service),
):
    """
    Generate candidate designs for a target objective using a trained engine.
    """
    config = GenerationConfig(
        dataset_name=request.dataset_name,
        target_objective=request.target_objective,
        solver_type=request.solver_type,
        version=request.version,
        n_samples=request.n_samples,
    )

    try:
        result = service.generate_candidates(config)
        return GenerateResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/engines", response_model=list[EngineListItem])
async def list_all_engines(
    dataset_name: str | None = None,
    service: InverseService = Depends(get_inverse_service),
):
    """
    List trained engines with optional dataset filter.
    """
    engines = service.list_engines(dataset_name)
    return [
        EngineListItem(
            dataset_name=e.get("dataset_name"),
            solver_type=e["solver_type"],
            version=e["version"],
            created_at=e["created_at"],
        )
        for e in engines
    ]


@router.post("/engines/delete")
async def delete_engines(
    request: BulkDeleteEnginesRequest,
    service: InverseService = Depends(get_inverse_service),
):
    """
    Delete one or multiple trained engines.
    """
    return service.delete_engines(request.engines)


@router.get("/engines/{dataset_name}", response_model=list[EngineListItem])
async def list_engines_for_dataset(
    dataset_name: str,
    service: InverseService = Depends(get_inverse_service),
):
    """
    List all trained engines for a specific dataset.
    """
    engines = service.list_engines(dataset_name)
    return [
        EngineListItem(
            dataset_name=e.get("dataset_name"),
            solver_type=e["solver_type"],
            version=e["version"],
            created_at=e["created_at"],
        )
        for e in engines
    ]
